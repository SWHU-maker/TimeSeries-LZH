import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Sequence, Optional, List


class WeatherSequenceDataset(Dataset):
    """
    基于 weather.csv 的简单滑窗序列数据集。
    - 使用所有数值列（除 date）或指定 feature_cols。
    - 每个样本做 per-window 标准化，以对齐 LZHModel 的 RevIN 处理。
    """

    def __init__(self, csv_path: str, seq_len: int, feature_cols: Optional[Sequence[str]] = None):
        super().__init__()
        df = pd.read_csv(csv_path)

        # 自动选择数值特征列
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != "date"]
        self.feature_cols = list(feature_cols)

        data = df[self.feature_cols].astype("float32")
        self.data = data.values  # (N, C)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len + 1)

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.seq_len]  # (L, C)
        x = torch.tensor(window, dtype=torch.float32)
        # per-window 标准化，避免分布漂移
        mean = x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mean) / std
        return x

