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

    # 训练
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    device: Optional[str] = None  # None 自动选择
    log_interval: int = 50
    save_dir: str = "checkpoints/diffusion_lzh"
    save_name: str = "diffusion_lzh.pt"

    # 扩散参数（与 LZHModel 默认保持一致）
    noise_schedule: str = "linear-var"
    noise_scale: float = 1.0
    noise_min: float = 1e-4
    noise_max: float = 0.02
    noise_steps: int = 100
    mean_type: str = "EPSILON"

