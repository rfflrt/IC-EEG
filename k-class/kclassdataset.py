import numpy as np
import torch
from torch.utils.data import Dataset

class KClassDataset(Dataset):
    def __init__(self, k, path):
        data = np.load(path, allow_pickle=True)
        self.X = np.asarray(data)
        N = len(self.X)
        part = N // k
        y = np.array([], dtype=np.int64)
        for i in range(k):
            y = np.concatenate([y, i * np.ones(part, dtype=np.int64)])
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        x = x.unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y