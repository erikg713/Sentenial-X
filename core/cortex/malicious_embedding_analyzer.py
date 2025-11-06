# core/cortex/malicious_embedding_analyzer.py
# A simple Python script to analyze embeddings for potential malicious characteristics.
# This is a demonstration script using anomaly detection on vector embeddings.
# Assumptions: Embeddings are numpy arrays or lists of floats.
# Uses Isolation Forest for detecting outliers, which could indicate malicious embeddings.

import numpy as np
from sklearn.ensemble import IsolationForest  # Note: If sklearn is not available, adapt with other methods.
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output.

class MaliciousEmbeddingAnalyzer:
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the analyzer.
        :param contamination: The proportion of outliers in the data set (malicious embeddings).
        :param random_state: Seed for reproducibility.
        """
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.fitted = False

    def fit(self, embeddings):
        """
        Fit the model on benign embeddings.
        :param embeddings: List or array of benign embeddings (numpy arrays or lists).
        """
        embeddings_array = self._to_numpy(embeddings)
        self.model.fit(embeddings_array)
        self.fitted = True
        print("Model fitted on benign embeddings.")

    def analyze(self, new_embeddings):
        """
        Analyze new embeddings for malicious characteristics.
        :param new_embeddings: List or array of embeddings to analyze.
        :return: List of booleans indicating if each embedding is malicious (True) or not.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted on benign embeddings first.")
        
        new_embeddings_array = self._to_numpy(new_embeddings)
        predictions = self.model.predict(new_embeddings_array)
        # -1 indicates anomaly (malicious), 1 indicates normal.
        is_malicious = [pred == -1 for pred in predictions]
        return is_malicious

    def _to_numpy(self, embeddings):
        """
        Convert list of embeddings to numpy array.
        """
        if isinstance(embeddings, list):
            return np.array(embeddings)
        elif isinstance(embeddings, np.ndarray):
            return embeddings
        else:
            raise ValueError("Embeddings must be a list or numpy array.")

# Example usage:
if __name__ == "__main__":
    # Sample benign embeddings (random for demo).
    benign_embeddings = np.random.rand(100, 128)  # 100 embeddings of dimension 128.
    
    # Sample test embeddings, including some potential outliers.
    test_embeddings = np.random.rand(10, 128)
    test_embeddings[5] += 10  # Make one an outlier.
    
    analyzer = MaliciousEmbeddingAnalyzer(contamination=0.05)
    analyzer.fit(benign_embeddings)
    results = analyzer.analyze(test_embeddings)
    
    print("Malicious detections:", results)
