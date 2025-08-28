"""
KMeans Clustering
Semantic clustering of events or embeddings.
"""

from sklearn.cluster import KMeans
import numpy as np

class KMeansCluster:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
