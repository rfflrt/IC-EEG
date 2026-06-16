import numpy as np
import torch
from torch.utils.data import Dataset

class KClassDataset(Dataset):
    def __init__(self, k, path):
        data = np.load(path, allow_pickle=True)
        self.X = torch.from_numpy(data).float()
        N = len(self.X)
        part = N // k
        y = []
        for i in range(k):
            y.extend([i] * part)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y