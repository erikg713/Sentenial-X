import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Simple text vectorizer for threat payloads
vectorizer = HashingVectorizer(
    n_features=512, alternate_sign=False, norm="l2", lowercase=True
)

def preprocess(payload: str) -> np.ndarray:
    """
    Normalize and vectorize input for ML model
    """
    clean = re.sub(r"[^a-zA-Z0-9 ]", " ", payload.lower())
    return vectorizer.transform([clean]).toarray()[0]
