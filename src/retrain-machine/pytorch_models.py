import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
import numpy as np

import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments
)

import os
from tqdm import tqdm
from collections import namedtuple
import joblib
import functools
import utils_data


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, labels=None, n_classes=None, features=None, max_length=128):
        """
        Args:
            texts: list of tweet strings
            labels: list of labels
            tokenizer: the tokenizer accomapnying the HF model
            features: numpy array of other features (float32)
        """
        self.max_length = max_length
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        if n_classes is not None and n_classes > 1:
            self.sample_weights = compute_sample_weights(labels)

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
        if self.features is not None:
            item["features"] = torch.tensor(self.features[idx])
        elif type(idx) != int:
            batch_shape = len(idx)
            item["features"] = torch.empty(batch_shape, 0)
        else:  # to maintain compatibility with the model input signature
            item["features"] = torch.empty(0,)
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.texts)


class CustomTrainer(Trainer):
    # subclassing the Trainer to inject custom behavior

    def __init__(self, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sample_weights = self.train_dataset.sample_weights
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        return DataLoader(
            self.train_dataset,
            shuffle=False,  # the sampler is already random
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    https://docs.pyro.ai/en/latest/_modules/pyro/contrib/cevae.html#FullyConnected
    """

    def __init__(self, sizes, final_activation=None):
        # sizes: first elem is the input size, last input is the output size
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)


class ViralityClassifier(nn.Module):
    # feel free to change the architecture

    def __init__(self, checkpoint, n_features=0, n_classes=4, freeze_bert=False, dropout=0.2, hidden_sizes=[]):
        super(ViralityClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.n_classes = n_classes

        if freeze_bert:
            self.freeze_bert()

        bert_hdim = self.bert.config.hidden_size
        sizes = [bert_hdim+n_features]
        sizes.extend(hidden_sizes)
        sizes.append(n_classes)

        self.dropout = nn.Dropout(dropout)
        self.classifier = FullyConnected(sizes)

    def forward(self, input_ids, token_type_ids, attention_mask, features, labels=None):
        """This method was coded with compatibility with HF Trainer API
        Basically, the args are ordered according to the items
        in the returned dict from Dataset above
        """
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # take the [CLS] output of BERT - can be interpreted as the summary of the sequence
        seq_summary = bert_outputs["last_hidden_state"][:, 0, :]
        seq_summary = self.dropout(seq_summary)
        logits = self.classifier(torch.cat([seq_summary, features], dim=1))
        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=self.compute_loss(logits, labels) if labels is not None else None,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions
        )

    def predict(self, *args, **kwargs):
        outputs = self.forward(*args, **kwargs)
        return F.softmax(outputs.logits, dim=1), outputs.logits.argmax(dim=1)


    def compute_loss(self, logits, labels):
        logits = logits.view(-1, self.n_classes)
        labels = labels.view(-1)
        if self.n_classes == 1:
            loss = F.mse_loss(logits.view(-1), labels.float())
        else:
            loss = F.cross_entropy(logits, labels)
        return loss

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_sample_weights(labels):
    class_sample_counts = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / class_sample_counts
    sample_weights = class_weights[labels]
    return sample_weights


def get_torch_dataset(dframe, feature_names, tokenizer, labels=None, n_classes=None):
    config = get_config()
    dset = TweetDataset(
        texts=dframe["tweet"].to_list(),
        tokenizer=tokenizer,
        labels=labels,
        n_classes=n_classes,
        features=utils_data.get_feature_array(dframe, feature_names),
        max_length=config.max_length
    )
    return dset


def get_torch_datasets(splits, feature_names, tokenizer, with_label=True, n_classes=None):
    datasets = {}
    for phase in splits.keys():
        datasets[phase] = get_torch_dataset(
            splits[phase],
            feature_names,
            tokenizer,
            splits[phase]["label"].to_list() if with_label else None,
            n_classes=n_classes
        )
    return datasets


def evaluate(eval_frame, features, n_classes, model, tokenizer, bsize=128, torchscript=False):
    eval_pred = compute_batch_predictions(
        eval_frame,
        features,
        model,
        tokenizer,
        bsize=bsize,
        torchscript=torchscript
    )
    metrics = compute_metrics(n_classes, eval_pred)
    print(metrics)
    return metrics


@torch.no_grad()
def compute_batch_predictions(eval_frame, features, model, tokenizer, bsize=128, torchscript=False):
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() and not torchscript else torch.device("cpu")
    model.to(device)
    y_true = eval_frame.label.to_list()
    dset = get_torch_dataset(
        eval_frame,
        features,
        tokenizer,
        labels=None if torchscript else y_true
    )

    n_batches = (len(eval_frame) + bsize - 1) // bsize
    val_loader = torch.utils.data.DataLoader(dset, batch_size=bsize)
    y_pred = []  # model predictions
    for batch in tqdm(val_loader, total=n_batches):
        for key, value in batch.items():
            batch[key] = value.to(device)
        if not torchscript:
            raw_preds = model(**batch).logits.cpu().numpy()
        else:
            _, raw_preds = model(*batch.values())
        y_pred.append(raw_preds)
    y_pred = np.concatenate(y_pred, axis=0)
    return namedtuple("Prediction", ["label_ids", "predictions"])(label_ids=np.array(y_true), predictions=y_pred)


def compute_metrics(n_classes, eval_pred, torchscript=False):
    """to be used with the Trainer in the evaluation phase.
    eval_pred: NamedTuple of raw predictions (returned by model.forward)
    returns a dict
    """
    y_true = eval_pred.label_ids
    y_pred = eval_pred.predictions
    return_dict = {}
    if n_classes > 1:
        y_pred = y_pred.argmax(axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        return_dict = {"accuracy": accuracy}
        return_dict["weighted_avg_precision"] = class_report["weighted avg"]["precision"]
        return_dict["weighted_avg_f1"] = class_report["weighted avg"]["f1-score"]
        if n_classes == 2:
            if not torchscript:
                true_class_probs = F.softmax(torch.tensor(eval_pred.predictions), dim=1)[:, 1].numpy()
            else:
                true_class_probs = eval_pred.predictions
            return_dict["roc_auc"] = roc_auc_score(y_true, true_class_probs)
        for k, v in class_report.items():
            try:
                return_dict[f"recall_{k}"] = v["recall"]
            except Exception:
                break
    else:
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return_dict = {"mae": mae, "mape": mape, "r2": r2}
    return return_dict


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.eval()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().cpu()
        self.model
        for param in self.model.parameters():
            param.requires_grad=False

    def forward(self, *args, **kwargs):
        # for inference.
        output_tuple = self.model(*args, **kwargs)
        return output_tuple.logits.argmax(dim=1), F.softmax(output_tuple.logits, dim=1)


def save_torchscript_model(model, filename, config):
    # https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#
    model = ModelWrapper(model)
    example_input = [
        torch.zeros(1, config.max_length, dtype=torch.long),
        torch.zeros(1, config.max_length, dtype=torch.long),
        torch.zeros(1, config.max_length, dtype=torch.long),
        torch.zeros(1, len(config.features))
    ]
    jitted_model = torch.jit.trace(model, example_input)
    jitted_model = torch.jit.freeze(jitted_model.eval())
    jitted_model.save(filename)
    print("Saved traced model as {}".format(filename))


def load_jitted_model(filename):
    return torch.jit.load(filename)


def get_config():
    config = {
        "hf_checkpoint": "vinai/bertweet-covid19-base-uncased",#"google/bert_uncased_L-2_H-128_A-2",
        "features":[
            'created_at_dayofweek', 'created_at_hour', 'is_quote_tweet',
            'hashtags_count', 'mentions_count', 'word_count',
            'photos_count', 'has_video', 'urls_count'
        ],
        "target_var": "likes_count",
        "labels": [0, 1, float("inf")],
        "should_normalize": False,
        "freeze_bert": True,
        "hidden_sizes": [64,64],
        "early_stopping_patience": 3,
        "batch_size": 128,
        "n_epochs": 20,
        "learning_rate": 5e-5,
        "weight_decay": 1e-2,
        # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
        "lr_scheduler_type": "cosine_with_restarts",
        "dropout": 0.3,
        "max_length": 128,
        "seed": 69,
        "dataset": "tweets_covid.json.xz"
    }
    config = namedtuple("Config", list(config.keys()))(**config)
    return config  # convert to a dict with config._asdict()


def main(dataset=None):
    """
    dataset: path to a preprocessed dataset (can be a URL)
    """
    config = get_config()
    n_classes = len(config.labels) - 1  # labels are determined in the preprocessing step
    seed_everything(config.seed)

    df = utils_data.load_data(dataset if dataset is not None else config.dataset)
    if "label" not in df.columns:
        utils_data.label_data(df, config.target_var, config.labels)
        print(f"Class distribution: {dict(df.label.value_counts())}")

    splits = utils_data.split_data(df, config.seed, regression=n_classes == 1)
    if config.should_normalize:
        scaler, splits = utils_data.normalize_data(
            splits,
            include_labels=n_classes == 1,
            append_text=True
        )
        joblib.dump(scaler, "saved_models/torch_scaler.joblib")

    tokenizer = AutoTokenizer.from_pretrained(config.hf_checkpoint, normalization=True)
    datasets = get_torch_datasets(splits, config.features, tokenizer, n_classes=n_classes)

    training_args = TrainingArguments(
        report_to="none",
        output_dir=f'./results/',
        overwrite_output_dir=True,
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=500,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type=config.lr_scheduler_type,
        eval_accumulation_steps=10,
        dataloader_num_workers=os.cpu_count(),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False
    )

    model = ViralityClassifier(
        config.hf_checkpoint,
        n_features=len(config.features),
        n_classes=n_classes,
        freeze_bert=config.freeze_bert,
        dropout=config.dropout,
        hidden_sizes=config.hidden_sizes
    )

    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_threshold=0
    )

    trainer_class = CustomTrainer if n_classes > 1 else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        tokenizer=tokenizer,
        compute_metrics=functools.partial(compute_metrics, n_classes),
        callbacks=[early_stopping]
    )

    trainer.evaluate()  # "sanity check"
    # the best model will be loaded at the end of the training
    trainer.train()

    # evaluate(splits["train"], config.features, n_classes, model, tokenizer)
    eval_results = evaluate(splits["test"], config.features, n_classes, model, tokenizer)

    save_torchscript_model(model, "saved_model.pt", config)
    
    return {"results": eval_results, "config": config._asdict(), "test_data": splits["test"]}

def run_training(dataset):
    result_data = main(dataset)
    dataset_name, _ = os.path.splitext(os.path.basename(dataset))
    test_data_path = f"{dataset_name}_test.json"
    utils_data.save_df_as_json(result_data["test_data"], test_data_path)
    del result_data["test_data"]
    result_data["config"]["dataset"] = dataset
    result_data["config"]["test_data_path"] = test_data_path
    return result_data

if __name__ == "__main__":
    main()
