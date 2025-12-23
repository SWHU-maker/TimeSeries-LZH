# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Sequence, Optional, List
import numpy as np


class WeatherSequenceDataset(Dataset):
    """
    基于 weather.csv 的简单滑窗序列数据集。
    修正：支持外部传入归一化参数，避免数据泄露。
    """

    def __init__(self, csv_path: str, seq_len: int, feature_cols: Optional[Sequence[str]] = None, 
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        super().__init__()
        df = pd.read_csv(csv_path)

        # 自动选择数值特征列
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != "date"]
        self.feature_cols = list(feature_cols)

        self.data = df[self.feature_cols].astype("float32").values  # (N, C)
        
        # 修正：如果提供了归一化参数则使用，否则从当前数据计算
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = self.data.mean(axis=0, keepdims=True)  # (1, C)
            self.std = self.data.std(axis=0, keepdims=True) + 1e-5  # (1, C)
        
        # 应用归一化
        self.data = (self.data - self.mean) / self.std
        
        self.seq_len = seq_len
        
        # 打印数据统计信息（仅在首次创建时打印）
        if mean is None and std is None:
            print(f"[dataset] Total samples: {len(self.data)}")
            print(f"[dataset] Feature columns ({len(self.feature_cols)}): {self.feature_cols}")
            print(f"[dataset] Normalized mean: {self.mean.mean():.4f}, std: {self.std.mean():.4f}")

    def __len__(self):
        return max(0, len(self.data) - self.seq_len + 1)

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.seq_len]  # (L, C)
        x = torch.tensor(window, dtype=torch.float32)
        return x
    
    def get_normalization_params(self):
        """返回归一化参数，用于其他数据集"""
        return self.mean, self.std