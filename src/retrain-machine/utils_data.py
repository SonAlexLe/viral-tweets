from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pandas as pd
import numpy as np

from emoji import demojize
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

import os, tempfile, urllib


def get_features(return_empty=False):
    features = [
        'created_at_dayofweek', 'created_at_hour', 'is_quote_tweet',
        'hashtags_count', 'mentions_count', 'word_count',
        'photos_count', 'has_video', 'urls_count'
    ] if not return_empty else []
    return features


def normalizeToken(token):
    # lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize_tweet(tweet):
    #https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    tokens = tweet_tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


def download_data(data_url, use_temp=True):
    json_name = None if use_temp else os.path.basename(data_url)
    data_path, _ = urllib.request.urlretrieve(url=data_url, filename=json_name)
    return data_path


def label_data(df, target, bins=None):
    """In-place operation
    bins=None means a regression target
    """
    if bins is not None:
        n_classes = len(bins) - 1
        df["label"] = pd.cut(df[target], bins=bins, include_lowest=True, right=False)
        new_labels = [i for i in range(n_classes)]
        df["label"] = df["label"].cat.rename_categories(new_labels)
        df["label"] = df["label"].astype("int")
    else:
        df["label"] = df[target].astype("float")
        n_classes = 1
    return n_classes


def preprocess(json_path, target_var="likes_count", bins=[0, 1, float("inf")]):
    data = pd.read_json(json_path, lines=True, orient="records")
    # drop tweets that are replies, duplicates, and non-English
    data["n_reply_to"] = data.reply_to.apply(lambda x: len(x) if isinstance(x, list) else 0)
    data = data[(data.n_reply_to == 0) & (data["language"] == "en")].drop_duplicates(subset="id").reset_index(drop=True)
    # figure out the counts for the non-text content
    data["photos_count"] = data["photos"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    data["has_video"] = data.apply(lambda x: x.video == 1 and x.photos_count == 0, axis=1)
    data["urls_count"] = data["urls"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    # transform features with lists to counts
    data["hashtags_count"] = data["hashtags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    data["mentions_count"] = data["mentions"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    data["is_quote_tweet"] = data["quote_url"].apply(lambda x: x != '')
    data["word_count"] = data["tweet"].apply(lambda x: len(x.split()))
    data["created_at_dayofweek"] = data["created_at"].dt.dayofweek
    data["created_at_hour"] = data["created_at"].dt.hour
    data["tweet"] = data["tweet"].apply(normalize_tweet)
    if target_var is not None:
        n_classes = label_data(data, target_var, bins)
    columns_to_keep = [
        "id", "created_at", "created_at_dayofweek", "created_at_hour",
        "username", "tweet",
        "photos_count", "has_video", "urls_count",
        "hashtags_count", "mentions_count",
        "is_quote_tweet",
        "word_count",
        "label"
    ]
    return data[columns_to_keep]


def save_df_as_json(df, filename=None):
    if filename is not None:
        df.to_json(filename, lines=True, orient="records", date_format="iso", date_unit="s")
        return filename
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as result_file:
        df.to_json(result_file, lines=True, orient="records", date_format="iso", date_unit="s")
        return result_file.name


def preprocess_pipeline(data_url, logger, use_temp=False, target_var="likes_count", bins=[0, 1, float("inf")]):
    logger.info(f"Downloading data from \"{data_url}\"...")
    data_path = download_data(data_url, use_temp=use_temp)
    logger.info(f"Data downloaded to {data_path}. Preprocessing...")
    processed_data = preprocess(data_path, target_var, bins)
    logger.info(f"There are {len(processed_data)} tweets in the preprocessed dataset.")
    processed_data_path = save_df_as_json(
        processed_data,
        filename=None if use_temp else data_path
    )
    logger.info(f"Preprocessed data saved to {processed_data_path}")
    if processed_data_path != data_path:
        os.unlink(data_path)  # no need for original data
    return processed_data_path


def load_data(filepath):
    df = pd.read_json(filepath, lines=True)
    return df


def get_feature_array(dframe, feature_names):
    if len(feature_names) > 0:
        feature_array = dframe[feature_names].astype("float32").to_numpy()
        return feature_array
    return None


def split_data(df, random_state, regression=False):
    splits = {}
    if regression:
        splits["train"], test_val = train_test_split(df, test_size=0.3, random_state=random_state)
        splits["val"], splits["test"] = train_test_split(test_val, test_size=0.5, random_state=random_state)
    else: # stratify (keeping class distribution)
        splits["train"], test_val = train_test_split(df, test_size=0.3, random_state=random_state, stratify=df.label)
        splits["val"], splits["test"] = train_test_split(test_val, test_size=0.5, random_state=random_state, stratify=test_val.label)
    for phase, v in splits.items():
        splits[phase] = v.reset_index(drop=True)
    return splits


def normalize_data(splits, include_labels=False, append_text=False):
    # Don't put raw text data here
    # include_labels: whether to preprocess the target variable as well (for regression)
    # not used
    features = get_features()
    if include_labels:
        features.append("label")

    scaler = preprocessing.MinMaxScaler().fit(splits["train"][features].to_numpy())

    data = {}
    phases = ["train", "val", "test"]
    for phase in phases:
        X = scaler.transform(splits[phase][features].to_numpy())
        if not include_labels:
            y = splits[phase][["label"]].to_numpy()
            data[phase] = np.hstack((X, y))
        else:  # X contains the label as the final column
            data[phase] = X

    if "label" not in features:
        features = features + ["label"]
    scaled_splits = {}
    for phase in phases:
        scaled_splits[phase] = pd.DataFrame(data[phase], columns=features)
        if not include_labels:
            scaled_splits[phase]["label"] = scaled_splits[phase]["label"].astype("int32")
        if append_text:
            scaled_splits[phase]["tweet"] = splits[phase]["tweet"]
            scaled_splits[phase] = scaled_splits[phase][["tweet"]+features]

    return scaler, scaled_splits


