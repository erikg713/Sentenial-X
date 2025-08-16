# sentenial-x/apps/ml_pipeline/utils.py
import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Logging Utilities
# -------------------------
def setup_logger(name="ml_pipeline", log_file="ml_pipeline.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

logger = setup_logger()

# -------------------------
# Data Preprocessing
# -------------------------
def normalize(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Normalize numeric features in a DataFrame.
    method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return df
    encoder = OneHotEncoder(sparse=False, drop="first")
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    df = df.drop(columns=cat_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

def handle_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Fill missing values in numeric columns.
    strategy: 'mean', 'median', or 'zero'
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "zero":
            df[col].fillna(0, inplace=True)
        else:
            raise ValueError("strategy must be 'mean', 'median', or 'zero'")
    return df

# -------------------------
# Model Persistence
# -------------------------
def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} does not exist")
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model

# -------------------------
# Evaluation Metrics
# -------------------------
def evaluate_classification(y_true, y_pred, average="binary"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()

def compute_roc_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)
