# coding : utf-8
# Author : Yuxiang Zeng
# TimeMixerConfig.py

# exp_model注册
# elif config.model == 'timemixer':
#             self.model = TimeMixer(
#                     input_size=config.input_size,
#                     config=config)

from dataclasses import dataclass
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class TimeMixerConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    """
    Configuration for TimeMixer baseline
    """
    # ===== Model Settings =====
    model: str = 'timemixer'
    
    # ===== Dataset Settings =====
    dataset: str = 'weather'  # weather forecasting
    ts_var: int = 1  # 21 variables for weather data
    
    # ===== Sequence Settings =====
    seq_len: int = 96  # input sequence length
    pred_len: int = 96  # prediction length
    
    # ===== Model Architecture =====
    d_model: int = 128  # model dimension
    d_ff: int = 256  # feedforward dimension (typically 2*d_model)
    e_layers: int = 2  # number of encoder layers (PDM blocks)
    
    # ===== Multi-Scale Settings =====
    down_sampling_layers: int = 3  # number of down-sampling layers
    down_sampling_window: int = 2  # down-sampling window size
    down_sampling_method: str = 'avg'  # 'avg', 'max', or 'conv'
    
    # ===== Decomposition Settings =====
    decomp_method: str = 'moving_avg'  # 'moving_avg' or 'dft_decomp'
    moving_avg: int = 25  # kernel size for moving average
    top_k: int = 5  # top-k frequencies for DFT decomposition
    
    # ===== Training Settings =====
    bs: int = 32  # batch size
    lr: float = 1e-3  # learning rate
    epochs: int = 100
    patience: int = 10  # early stopping patience
    dropout: float = 0.1
    
    # ===== Other Settings =====
    rank: int = 64  # not used in TimeMixer but kept for compatibility
    use_amp: bool = False  # mixed precision training
    verbose: int = 1
    
    # ===== Regularization =====
    decay: float = 0.0  # weight decay
    
    # ===== Monitoring =====
    monitor_metric: str = 'MAE'  # metric to monitor for early stopping