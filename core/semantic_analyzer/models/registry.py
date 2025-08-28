"""
Central registry of all semantic analyzer models.
Maps model names to their full import paths for dynamic loading.
"""

MODEL_REGISTRY = {
    # Embeddings
    "bert_embedder": "core.semantic_analyzer.models.embeddings.bert.BertEmbedder",
    # Classifiers
    "svm_classifier": "core.semantic_analyzer.models.classifiers.svm.SVMClassifier",
    # Clustering
    "kmeans_cluster": "core.semantic_analyzer.models.clustering.kmeans.KMeansCluster",
}
