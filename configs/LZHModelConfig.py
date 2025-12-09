# LZHModelConfig.py

# coding: utf-8
# Author: mkw (adapted for LZHModel by AI)
# Date: 2025-12-01 15:45
# Description: LZHModelConfig

# 假设这些是你的配置文件基类
from configs.default_config import *
from configs.MainConfig import OtherConfig
from dataclasses import dataclass


@dataclass
class LZHModelConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    # --- 框架运行所需的必要属性 (从基类或默认设置中补齐) ---
    debug: bool = False             # 解决 'AttributeError: ... has no attribute 'debug''
    classification: bool = False    # 解决 'AttributeError: ... has no attribute 'classification''
    # ----------------------------------------------------------------------
    
    # 核心配置
    model: str = 'lzhmodel'
    
    # 模型架构配置
    d_model: int = 128
    num_layers: int = 3
    n_heads: int = 8
    revin: bool = True
    
    # 训练配置
    bs: int = 96
    epochs: int = 20
    patience: int = 3
    verbose: int = 1
    dropout: float = 0.1
    amp: bool = False
    
    # 扩散模型配置
    # use_diff=0: 不创建也不使用扩散模型，forward 直接返回原输入
    # use_diff=1: 创建并使用扩散模型
    #   - 如果提供了 diffusion_ckpt: 加载预训练模型并冻结参数，仅用于推理
    #   - 如果没有 diffusion_ckpt: 初始化新模型，跟随训练
    use_diff: int = 1
    noise_scale: float = 1.0
    noise_steps: int = 40
    lamda: float = 0.5  # 用于控制扩散损失的权重
    diffusion_ckpt = "checkpoints/diffusion_lzh/diffusion_lzh.pt"
    
    rounds: int = 3