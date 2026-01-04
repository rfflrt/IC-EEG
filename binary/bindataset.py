import numpy as np
import torch
from torch.utils.data import Dataset

class BinEEGDataset(Dataset):
    """
    Loads .npy of shape (N,100,17). Assumes first half are positives (label=1), second half negatives (label=0).
    Returns x with shape (1,100,17) and y as long scalar (0/1).
    """
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.X = np.asarray(data)
        if self.X.ndim != 3:
            raise ValueError(f"Expected file with shape (N,100,17), got {self.X.shape}")
        N = len(self.X)
        half = N // 2
        self.y = np.concatenate([np.ones(half, dtype=np.int64), np.zeros(N - half, dtype=np.int64)])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        x = x.unsqueeze(0)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y