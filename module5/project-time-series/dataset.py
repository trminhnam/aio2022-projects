import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from utils import sliding_window


class TelsaStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader
    from utils import sliding_window

    df = pd.read_csv("./data/tesla-stock-price.csv")
    df.head()

    if "date" in df.columns:
        df.drop(["date"], axis=1, inplace=True)
    target_idx = df.columns.get_loc("close")
    print(f"Target index: {target_idx}")

    for col in df.columns:
        df[col] = df[col].replace(",", "", regex=True).astype(float)

    mean = df.mean()
    std = df.std()

    # normalize data
    df = df.astype(float)
    df = (df - mean) / std
    df.head()

    data = sliding_window(df, 11, 1)
    data = np.array(data)

    X = data[:, :-1, :]
    y = data[:, -1, target_idx : target_idx + 1]
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")

    dataset = TelsaStockDataset(X, y)
    print(f"len(dataset): {len(dataset)}")
    print(f"dataset[0]: {dataset[0]}")
