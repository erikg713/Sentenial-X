#!/usr/bin/env python3
# scripts/train_models.py
# Train & Export Sentenial-X Models
# Usage: python scripts/train_models.py

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

def train_isolation_forest():
    # Generate sample data (normal + anomalies)
    rng = np.random.RandomState(42)
    X_normal = 0.3 * rng.randn(1000, 7)  # 7 features from extract_features_fast + toxicity
    X_anomalies = rng.uniform(low=-4, high=4, size=(50, 7))
    X = np.concatenate([X_normal, X_anomalies], axis=0)

    # Train
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)

    # Save .pkl
    joblib.dump(model, 'models/isolation_forest.pkl')
    print("Saved isolation_forest.pkl")

    # Export ONNX (with scores)
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = to_onnx(model, X.astype(np.float32), initial_types=initial_type,
                         options={id(model): {'score_samples': True}},
                         target_opset=12)
    with open('models/isolation_forest.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("Exported isolation_forest.onnx")

if __name__ == "__main__":
    train_isolation_forest()
