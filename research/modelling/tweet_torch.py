import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import wandb

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification
)

# define features in addition to the tweet text
# set this as an empty list [] to only use the tweet text
features = [
    'created_at_dayofweek', 'created_at_hour',
    'hashtags_count', 'mentions_count',
    'cashtags_count', 'is_quote_tweet'
]
# features = []

wandb.init(  # addtitional hyperparameters will be defined below
    config={
        "hf_checkpoint": "google/bert_uncased_L-2_H-128_A-2",
        "freeze_bert": True,
        "features": features,
        "early_stopping_patience": 3,
        "dropout": 0.2,
        "max_length": 128,
        "dataset": "https://users.aalto.fi/~les1/all_tweets.json.xz"
    },
    name="no0retweets-128hdim-newclassifier-frozen",
    group="all-3labels",
    entity="tweet_virality",
    save_code=True,
    project="ml-dev"
)


training_args = TrainingArguments(
    report_to="wandb",
    output_dir=f'./results/{wandb.run.name}',                    # output directory
    overwrite_output_dir=True,
    num_train_epochs=20,                       # total number of training epochs
    per_device_train_batch_size=128,           # batch size per device during training (smaller size for bigger models otherwise out of memory)
    per_device_eval_batch_size=128,            # batch size for evaluation
    warmup_steps=1000,                          # number of warmup steps for learning rate scheduler
    learning_rate=1e-3,
    weight_decay=1e-3,                         # strength of weight decay
    logging_steps=50,                          # logs after every x steps
    evaluation_strategy="epoch",               # when to evaluate
    save_strategy="epoch",                     # when to save
    eval_accumulation_steps=10,
    dataloader_num_workers=os.cpu_count(),  # change this to run faster (might not work on some machines)
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True                     # for metric_for_best_model
)

# save whatever you want with this
code = wandb.Artifact('training_script', type='code')
code.add_file("tweet_torch.py")
wandb.run.log_artifact(code)

df = pd.read_json(wandb.config.dataset, lines=True)
df = df.loc[df.retweets_count > 0, :]
df["label"] = pd.cut(df["retweets_count"], bins=[1,10,100,float("inf")], include_lowest=True, right=False)
df["label"] = df["label"].cat.rename_categories([0,1,2])

splits = {}
checkpoint = wandb.config.hf_checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint, normalization=True)
splits["train"], test_val = train_test_split(df, test_size=0.3, random_state=69, stratify=df.label)
splits["val"], splits["test"] = train_test_split(test_val, test_size=0.5, random_state=69, stratify=test_val.label)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, features=None, max_length=128):
        """
        Args:
            texts: list of tweet strings
            labels: list of labels
            tokenizer: the tokenizer accomapnying the HF model
            features: pd.DataFrame of other features
        """
        self.max_length = max_length
        self.texts = texts
        self.features = features.values.astype("int32") if features is not None else None
        self.labels = labels
        self.tokenizer = tokenizer
        self.sample_weights = self.compute_sample_weights(labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        encodings = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def compute_sample_weights(self, labels):
        class_sample_counts = np.unique(labels, return_counts=True)[1]
        class_weights = 1. / class_sample_counts
        sample_weights = class_weights[labels]
        return sample_weights

datasets = {}
for split_name in splits.keys():
    datasets[split_name] = TweetDataset(
        texts=splits[split_name].tweet.to_list(),
        labels=splits[split_name].label.to_list(),
        tokenizer=tokenizer,
        features=None if len(features) == 0 else splits[split_name][features],
        max_length=wandb.config.max_length
    )

class ViralityClassifier(torch.nn.Module):
    """Feel free to change the architecture
    This class knows whether there are features besides the tweet body
    (just supply the corresponding n_features)
    https://github.com/The-AI-Summer/Hugging_Face_tutorials/blob/master/vit.py
    """
    def __init__(self, checkpoint, n_features=0, n_labels=4, freeze_bert=False, dropout=0.2):
        super(ViralityClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.n_labels = n_labels

        if freeze_bert:
            self.freeze_bert()

        bert_hdim = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Sequential(
	    torch.nn.Linear(bert_hdim+n_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_labels)
        )

    def forward(self, input_ids, token_type_ids, attention_mask, labels, features=None):
        """This method was coded with compatibility with HF Trainer API
        Basically, the args are ordered according to the items
        in the returned dict from Dataset above
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # takes the [CLS] output of BERT - can be interpreted as the summary of the sequence
        seq_summary = bert_outputs["last_hidden_state"][:, 0, :]
        seq_summary = self.dropout(seq_summary)
        logits = self.classifier(torch.cat([seq_summary, features], dim=1))

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=self.compute_loss(logits, labels),
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions
        )
    
    def compute_loss(self, logits, labels):
        logits = logits.view(-1, self.n_labels)
        labels = labels.view(-1)
        if self.n_labels == 1:
            # loss = torch.nn.functional.poisson_nll_loss(logits, labels)  # this doesn't work for now
            loss = torch.nn.functional.mse_loss(logits.view(-1), labels.float())
        elif self.n_labels == 2:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False


def compute_metric(eval_pred):
    # to be used with the Trainer in the evaluation phase.
    # Takes in a dict (or NamedTuple, not sure). Must return a dict.
    return {"accuracy": (eval_pred.label_ids == eval_pred.predictions.argmax(-1)).mean()}


model = ViralityClassifier(
    checkpoint,
    n_features=len(features),
    n_labels=3,
    freeze_bert=wandb.config.freeze_bert,
    dropout=wandb.config.dropout
)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)

early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=wandb.config.early_stopping_patience,
    early_stopping_threshold=0
)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sample_weights = self.train_dataset.sample_weights
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

trainer = CustomTrainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=datasets["train"],     # training dataset
    eval_dataset=datasets["val"],        # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metric,      # the compute_metric function defined above
    callbacks=[early_stopping]
)

trainer.evaluate()  # should be no better than 25% accuracy since we have 4 classes

trainer.train()

best_model_artifact = wandb.Artifact(
    "TinyBertweetClassifier",
    type="model",
    description="best TinyBert model for virality prediction"
)
best_ckpt = np.min([int(fol.split("-")[1]) for fol in os.listdir(f"./results/{wandb.run.name}")])
best_model_artifact.add_dir(f"./results/{wandb.run.name}/checkpoint-{best_ckpt}")
wandb.run.log_artifact(best_model_artifact)


def evaluate(eval_set, features, model, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    dset = TweetDataset(
        texts=eval_set.tweet.to_list(),
        labels=eval_set.label.to_list(),
        tokenizer=tokenizer,
        features=None if len(features) == 0 else eval_set[features],
        max_length=wandb.config.max_length
    )
    val_loader = torch.utils.data.DataLoader(dset, batch_size=64)
    losses = []
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            for key, value in batch.items():
                batch[key] = value.to(device)
            outputs = model(**batch)
            preds.extend(outputs.logits.argmax(dim=1).cpu().numpy().tolist()) 
            losses.append(outputs.loss.item())
    loss = np.mean(losses)
    print("Mean loss: {:.4f}".format(loss))
    acc = accuracy_score(eval_set.label.to_list(), preds)
    print("Accuracy: {:.4f}".format(acc))
    conf_mat = confusion_matrix(eval_set.label.to_list(), preds, normalize="true")  # C_ij = true class i, pred class j
    print(conf_mat)
    conf_df = pd.DataFrame(data=conf_mat)
    conf_table = wandb.Table(dataframe=conf_df)
    wandb.run.log({"confusion_matrix": conf_table})
    wandb.run.summary["loss"] = loss
    wandb.run.summary["best_accuracy"] = acc

evaluate(splits["val"], features, model, tokenizer)

