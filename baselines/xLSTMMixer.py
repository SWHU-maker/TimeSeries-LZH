# file: baselines/xLSTMMixer.py
# coding: utf-8
# Author: migrated xLSTM-Mixer baseline for your framework (closer to paper)

import torch
from torch import nn
from einops import rearrange

from layers.revin import RevIN


class NLinear(nn.Module):
    """
    初始线性预测（Initial Linear Forecast）
    对每个通道独立地在时间维上做线性变换：
        x: [B, L, D] -> y: [B, H, D]
    其中 H = pred_len
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        # 对时间维做线性映射：L -> H，所有通道共享同一个线性层
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = rearrange(x, "b l d -> b d l")    # [B, D, L]
        y = self.linear(x)                    # [B, D, H]
        y = rearrange(y, "b d h -> b h d")    # [B, H, D]
        return y


class xLSTMMixerBaseline(nn.Module):
    """
    xLSTM-Mixer 风格的实现（简化版，用普通 LSTM 近似 xLSTM block）

    输入:
        x:      [B, seq_len, D]
        x_mark: [B, seq_len, *]   （这里暂时不用，只保持接口）

    输出:
        y:      [B, pred_len, D]

    结构对齐论文的思路：
    1) RevIN 标准化（可选）
    2) NLinear 做一个共享 across variates 的初始线性预测 y0
    3) 把 y0 视为 "每个变量有一个长度为 pred_len 的未来轨迹":
         y0: [B, H, D] -> [B, D, H]
       对每个变量的未来轨迹做上投影 H -> d_model 得到 embedding
    4) 在“变量维 D 上”跑 LSTM（代替原文的 xLSTM scalar memories）
    5) 做 multi-view：正向变量顺序 + 反向变量顺序 两个 view 的输出拼接
    6) 用线性层把两种 view 融合并映射回 [B, D, H]，再 residual 加到 y0 上
    7) RevIN 反标准化
    """

    def __init__(self, enc_in: int, config):
        """
        enc_in: 输入通道数 D（由 DataModule / DatasetInfo 决定）
        """
        super().__init__()
        self.config = config

        self.seq_len = config.seq_len       # L
        self.pred_len = config.pred_len     # H
        self.d_model = config.d_model       # 隐藏维度（论文里通常 256）
        self.revin = getattr(config, "revin", True)

        # ============= RevIN =============
        if self.revin:
            self.revin_layer = RevIN(
                num_features=enc_in, affine=False, subtract_last=False
            )

        # ============= 1) 初始线性预测（time mixing） =============
        self.nlinear = NLinear(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=enc_in,
        )

        # ============= 2) 上投影：每个变量的未来轨迹 H -> d_model =============
        # y0: [B, H, D] -> [B, D, H] 之后，对最后一维 H 做线性变换
        self.fc_up = nn.Linear(self.pred_len, self.d_model)

        # ============= 3) 在变量维 D 上跑 LSTM（近似 xLSTM block） =============
        self.num_layers = getattr(config, "num_layers", 3)  # 论文里多层 stack
        self.lstm = nn.LSTM(
            input_size=self.d_model,      # 每个变量的 embedding 维度
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,             # 输入形状 [B, D, d_model]
            bidirectional=False,
        )

        # ============= 4) multi-view + view mixing =============
        self.use_reverse_view = getattr(config, "use_reverse_view", True)
        in_dim_view = self.d_model * (2 if self.use_reverse_view else 1)

        # 把 multi-view 的输出映射回 “未来轨迹” 维度 H
        # 之后会 reshape 为 [B, H, D] 与 y0 做 residual
        self.fc_view = nn.Linear(in_dim_view, self.pred_len)

        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor):
        """
        x:      [B, L, D]
        x_mark: [B, L, *] 时间标记，目前不使用
        """
        # ---- (1) RevIN 标准化 ----
        if self.revin:
            x = self.revin_layer(x, "norm")

        # ---- (2) NLinear 初始线性预测: [B, L, D] -> [B, H, D] ----
        y0 = self.nlinear(x)          # [B, pred_len, D]

        # ---- (3) 把每个变量的未来轨迹上投影到 embedding ----
        # y0: [B, H, D] -> [B, D, H]
        y0_var_first = rearrange(y0, "b h d -> b d h")   # [B, D, H]
        # 对最后一维 H 做线性映射到 d_model
        h0 = self.fc_up(y0_var_first)                   # [B, D, d_model]
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        # ---- (4) 在变量维 D 上跑 LSTM（joint mixing over variates）----
        # 正向 view：变量顺序不变
        out_fwd, _ = self.lstm(h0)                      # [B, D, d_model]

        if self.use_reverse_view:
            # 反向 view：把变量顺序翻转
            h_rev = torch.flip(h0, dims=[1])            # [B, D, d_model]
            out_rev, _ = self.lstm(h_rev)               # [B, D, d_model]
            out_rev = torch.flip(out_rev, dims=[1])     # 再翻回来 [B, D, d_model]

            h_cat = torch.cat([out_fwd, out_rev], dim=-1)   # [B, D, 2*d_model]
        else:
            h_cat = out_fwd                              # [B, D, d_model]

        h_cat = self.act(h_cat)
        h_cat = self.dropout(h_cat)

        # ---- (5) view mixing + 映射回未来轨迹维度 H ----
        # [B, D, *] -> [B, D, H]
        y_delta = self.fc_view(h_cat)                   # [B, D, H]

        # [B, D, H] -> [B, H, D]，方便与 y0 对齐
        y_delta = rearrange(y_delta, "b d h -> b h d")  # [B, H, D]

        # ---- (6) residual refine：在初始线性预测 y0 的基础上加一个残差 ----
        y = y0 + y_delta                                # [B, H, D]

        # ---- (7) RevIN 反标准化 ----
        if self.revin:
            y = self.revin_layer(y, "denorm")

        # 最终输出 shape: [B, pred_len, D]
        return y
