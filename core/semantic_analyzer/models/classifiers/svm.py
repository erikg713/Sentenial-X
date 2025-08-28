"""
SVM Classifier
A simple example classifier for semantic event categories.
"""

from sklearn.svm import SVC
import numpy as np

class SVMClassifier:
    def __init__(self):
        self.model = SVC(probability=True)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)
