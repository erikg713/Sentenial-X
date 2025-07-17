from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(
    texts: List[str],
    labels: List[str]
) -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Vectorize input texts with TF-IDF and train a Logistic Regression classifier.
    Returns the trained model and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(texts, labels, random_state=42):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(random_state=random_state)
    model.fit(X, labels)
    return model, vectorizer
