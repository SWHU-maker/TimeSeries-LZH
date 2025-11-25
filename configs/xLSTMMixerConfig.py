# coding: utf-8
# Author: xLSTM-Mixer Config

from dataclasses import dataclass
from configs.default_config import *
from configs.MainConfig import OtherConfig


@dataclass
class xLSTMMixerConfig(
    ExperimentConfig,
    BaseModelConfig,
    LoggerConfig,
    DatasetInfo,
    TrainingConfig,
    OtherConfig,
):


    # --- 模型标识 ---
    model: str = "xlstm_mixer"

    # --- 训练超参数（可按论文/你现有习惯再调） ---
    bs: int = 64             # batch size for LTSF
    lr: float = 1e-3          # 初始学习率（论文级别通常 1e-3 ~ 3e-4）
    epochs: int = 200
    patience: int = 20        # ReduceLROnPlateau & EarlyStopping 使用

    # --- 结构相关超参数 ---
    # input_size: 由 DatasetInfo / DataModule 自动决定
    d_model: int = 256        # xLSTM hidden dim（论文一般用中等宽度）
    num_layers: int = 3       # LSTM 层数，对应“xLSTM block 堆叠层数”
    dropout: float = 0.1
    revin: bool = True        # 是否启用 RevIN

    # 是否使用反向视角 view（论文里的 multi-view mixing）
    use_reverse_view: bool = True

    # --- LTSF 常用窗口参数（你也可以在不同实验里覆盖） ---
    seq_len: int = 96         # history length
    pred_len: int = 96        # forecasting horizon

    # --- 占位符/默认值（为了兼容你的框架） ---
    rank: int = 32            # 虽然 xLSTM 不用 rank，但你的框架可能会访问
    weight_decay: float = 1e-4
