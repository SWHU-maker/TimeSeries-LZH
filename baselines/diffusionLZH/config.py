# config.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DiffusionLZHConfig:
    # 数据
    data_path: str = "datasets/weather.csv"
    seq_len: int = 96
    feature_cols: Optional[List[str]] = None  # None 表示自动选取非日期列
    train_split: float = 0.7
    val_split: float = 0.2  # 剩余自动为测试
    batch_size: int = 64
    num_workers: int = 0

    # 训练（关键修复：降低学习率，增加patience）
    epochs: int = 200
    lr: float = 5e-5  # 从 1e-4 降低到 5e-5，更稳定
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    device: Optional[str] = None  # None 自动选择
    log_interval: int = 50
    save_dir: str = "checkpoints/diffusion_lzh"
    save_name: str = "diffusion_lzh.pt"

    # 扩散参数
    noise_schedule: str = "linear-var"
    noise_scale: float = 1.0
    noise_min: float = 1e-4
    noise_max: float = 0.02
    noise_steps: int = 100
    mean_type: str = "EPSILON"
    
    # Early stopping 参数
    patience: int = 20  # 增加到 20，更宽容
    min_epochs: int = 10  # 新增：至少训练10个epoch