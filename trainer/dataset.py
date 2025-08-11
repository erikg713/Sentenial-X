# trainer/dataset.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int = 10, scaler: StandardScaler | None = None):
        """
        df: DataFrame with numeric features (timestamp sorted).
        window: number of timesteps per sample (for time-windowed autoencoder)
        """
        self.window = window
        self.features = df.select_dtypes(include=[float, int]).values
        if scaler is None:
            self.scaler = StandardScaler()
            # fit on flattened windows
            self.scaler.fit(self.features)
        else:
            self.scaler = scaler
        self.features = self.scaler.transform(self.features)
        self.samples = []
        for i in range(len(self.features) - window + 1):
            self.samples.append(self.features[i : i + window])
        self.samples = np.stack(self.samples).astype("float32")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]