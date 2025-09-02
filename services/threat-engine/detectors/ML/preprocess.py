import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def preprocess_input(data: dict) -> np.ndarray:
    """
    Convert raw threat input into a normalized feature vector.
    Expects data dict with numeric features.
    """
    features = [data.get(f"feature_{i}", 0.0) for i in range(1, 129)]
    features_array = np.array(features, dtype=np.float32).reshape(1, -1)
    features_scaled = scaler.fit_transform(features_array)
    return features_scaled
