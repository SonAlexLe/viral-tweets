# -*- coding: utf-8 -*-

# define features in addition to the tweet text
# set this as an empty list [] to only use the tweet text
features = [
    'created_at_dayofweek', 'created_at_hour',
    'hashtags_count', 'mentions_count',
    'cashtags_count', 'is_quote_tweet'
]

import pandas as pd
import torch
import numpy as np
import os
import transformers
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, Trainer, TrainingArguments

wandb.init(  # addtitional hyperparameters will be defined below
    config={
        "hf_checkpoint": "google/bert_uncased_L-2_H-128_A-2",
        "freeze_bert": False,
        "features": features,
        "early_stopping_patience": 5,
        "dataset": "https://users.aalto.fi/les1/tweets_verified.json.xz"
    },
    name="test-wandb5",
    entity="tweet_virality",
    project="ml-dev"
)

df = pd.read_json(wandb.config.dataset, lines=True)
df["label"] = pd.cut(df["retweets_count"], bins=[0,1,10,100,float("inf")], include_lowest=True, right=False)
df["label"].cat.rename_categories([0,1,2,3], inplace=True)

splits = {}
checkpoint = wandb.config.hf_checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint, normalization=True)
splits["train"], test_val = train_test_split(df, test_size=0.3, random_state=69, stratify=df.label)
splits["val"], splits["test"] = train_test_split(test_val, test_size=0.5, random_state=69, stratify=test_val.label)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, features=None):
        """
        Args:
            texts: list of tweet strings
            labels: list of labels
            tokenizer: the tokenizer accomapnying the HF model
            features: pd.DataFrame of other features
        """
        self.texts = texts
        self.features = features.values.astype("int32")
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        encodings = self.tokenizer(
            self.texts[idx],
            max_length=128,
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

datasets = {}
for split_name in splits.keys():
    datasets[split_name] = TweetDataset(
        texts=splits[split_name].tweet.to_list(),
        labels=splits[split_name].label.to_list(),
        tokenizer=tokenizer,
        features=None if len(features) == 0 else splits[split_name][features]
    )

class ViralityClassifier(torch.nn.Module):
    """Feel free to change the architecture 🤗
    This class knows whether there are features besides the tweet body
    (just supply the corresponding n_features)
    https://github.com/The-AI-Summer/Hugging_Face_tutorials/blob/master/vit.py
    """
    def __init__(self, checkpoint, n_features=0, n_labels=4, freeze_bert=False):
        super(ViralityClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.n_labels = n_labels

        if freeze_bert:
            self.freeze_bert()

        bert_hdim = self.bert.config.hidden_size
        if n_features == 0:
            n_features = n_labels
            self.classifier = None
        else:
            # processes (downscaled) bert CLS outputs and other features
            # input is a tensor where half of the dimensions are from BERT
            # and the remaining half are the additional features
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(n_features*2, n_features),
                torch.nn.ReLU(),
                torch.nn.Linear(n_features, n_labels)
            )

        # reduces the hidden dimension 
        # down to either the number of features
        # or the number of classes if there are no features
        self.bert_processor = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(bert_hdim, bert_hdim//3),
            torch.nn.ReLU(),
            torch.nn.Linear(bert_hdim//3, n_features)
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
        logits = self.bert_processor(bert_outputs["last_hidden_state"][:, 0, :])
        if self.classifier is not None:
            logits = self.classifier(torch.cat([logits, features], dim=1))

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
    return {"val_acc": (eval_pred.label_ids == eval_pred.predictions.argmax(-1)).mean()}

training_args = TrainingArguments(
    report_to="wandb",
    output_dir='./results',                    # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,                        # total number of training epochs
    per_device_train_batch_size=128,           # batch size per device during training (smaller size for bigger models otherwise out of memory)
    per_device_eval_batch_size=128,            # batch size for evaluation
    warmup_steps=500,                          # number of warmup steps for learning rate scheduler
    learning_rate=1e-4,
    weight_decay=1e-3,                         # strength of weight decay
    max_steps=10,                              # comment this out or change this
    logging_steps=10,                          # logs after every x steps
    evaluation_strategy="epoch",               # when to evaluate
    save_strategy="epoch",                     # when to save
    eval_steps=1000,                           # evaluation is done and logged every eval_steps. not used if evaluation_straty="epoch"
    save_steps=1000,                           # saves the model every save_steps
    eval_accumulation_steps=10,
    dataloader_num_workers=os.cpu_count(),  # change this to run faster (might not work on some machines)
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True                     # for metric_for_best_model
)

model = ViralityClassifier(
    checkpoint,
    n_features=len(features),
    n_labels=4,
    freeze_bert=wandb.config.freeze_bert
)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)

early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=wandb.config.early_stopping_patience,
    early_stopping_threshold=0
)

trainer = Trainer(
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