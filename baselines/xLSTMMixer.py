# file: baselines/xLSTMMixer.py
# coding: utf-8
# Author: xLSTM-Mixer style baseline adapted to your framework
#
# 接口保持与你现有模板一致：
#   - __init__(enc_in: int, config)
#   - forward(x: [B, L, D], x_mark: [B, L, ...]) -> [B, pred_len, D]
#
# 结构对齐论文/官方实现的“标准 xLSTM-Mixer”思路：
#   1) RevIN 标准化
#   2) NLinear 在时间维做线性预测 (带最后一个点的残差技巧)
#   3) 把预测结果视作每个变量的一条长度为 pred_len 的轨迹
#      [B, H, D] -> [B, D, H]
#   4) 对每个变量的轨迹用 Linear(H -> d_model) 做 embedding
#   5) 在“变量维 D 上”跑 LSTM stack（近似 xLSTM block）
#   6) 前向 view + 反向 view 拼接 (backcast)
#   7) Linear(d_model*2 -> pred_len)，映射回时间维 [B, D, H]
#   8) 还原回 [B, H, D]，RevIN 反标准化，输出 [B, pred_len, D]

import torch
from torch import nn
from einops import rearrange

from layers.revin import RevIN


class xLSTMMixerBaseline(nn.Module):
    def __init__(self, enc_in: int, config):
        """
        enc_in: 输入通道数 D
        config: 你的统一配置对象，需要包含：
            - seq_len : 输入长度 L
            - pred_len: 预测长度 H
            - d_model : 变量方向的 embedding / hidden 维度
            - num_layers: LSTM 堆叠层数（默认 3）
            - dropout : dropout 概率（默认 0.1）
            - revin   : 是否使用 RevIN（默认 True）
        """
        super().__init__()
        self.config = config

        self.seq_len = config.seq_len       # L
        self.pred_len = config.pred_len     # H
        self.enc_in = enc_in                # D
        self.d_model = config.d_model       # embedding / hidden dim
        self.revin = getattr(config, "revin", True)

        # === RevIN 标准化层 ===
        if self.revin:
            # 这里用的是你自己项目里的 RevIN 实现
            self.revin_layer = RevIN(
                num_features=enc_in, affine=False, subtract_last=False
            )

        # === NLinear backbone（时间方向线性映射）===
        # 对 x_enc 的「去 level」版本在线性层里做 L -> H，再加回最后一个点
        # x: [B, L, D] -> permute -> Linear(L -> H) -> [B, H, D]
        self.nlinear = nn.Linear(self.seq_len, self.pred_len)

        # === 变量方向 embedding：每个变量的未来轨迹 H -> d_model ===
        # 输入： [B, D, H]  对 H 这维做线性映射
        self.pre_encoding = nn.Linear(self.pred_len, self.d_model)

        # === 在变量维 D 上跑 LSTM（近似 xLSTM block）===
        # 输入输出都是 [B, D, d_model]
        self.num_layers = getattr(config, "num_layers", 3)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,   # 序列维度在 dim=1，这里是变量维 D
            bidirectional=False,
        )

        # 是否使用 backcast（即前后两个 view）
        self.backcast = True  # 按论文的最佳设置，直接固定为 True

        # backcast: concat(forward, backward)，所以维度翻倍
        in_dim_view = self.d_model * (2 if self.backcast else 1)

        # 把 multi-view 表征映射回时间维长度 H
        # 输入 [B, D, in_dim_view] -> 输出 [B, D, H]
        self.fc = nn.Linear(in_dim_view, self.pred_len)

        self.dropout = nn.Dropout(getattr(config, "dropout", 0.1))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor):
        """
        x:      [B, L, D]
        x_mark: [B, L, *]  （这里暂时不使用，只是为了接口兼容）
        返回:
        y:      [B, pred_len, D]
        """
        # ========= 1. RevIN 标准化 =========
        if self.revin:
            x = self.revin_layer(x, "norm")   # [B, L, D]

        # ========= 2. NLinear 时间方向预测 (官方做法) =========
        # 官方代码逻辑：
        #   seq_last = x[:, -1:, :]
        #   x_centered = x - seq_last
        #   x_centered -> permute to [B, D, L]
        #   Linear(L -> H)
        #   permute back to [B, H, D]
        #   x_pre_forecast = x_nlinear + seq_last
        seq_last = x[:, -1:, :].detach()          # [B, 1, D]
        x_centered = x - seq_last                 # 去掉 level

        # 时间维放到最后，便于线性层处理
        x_centered = x_centered.permute(0, 2, 1)  # [B, D, L]
        x_nlinear = self.nlinear(x_centered)      # [B, D, H]
        x_nlinear = x_nlinear.permute(0, 2, 1)    # [B, H, D]

        # 加回最后一个点，得到初始的时间方向预测
        x_pre_forecast = x_nlinear + seq_last     # [B, H, D]

        # ========= 3. 变量维视角 + embedding =========
        # 我们把每个变量看作一条长度为 H 的序列：
        # [B, H, D] -> [B, D, H]
        var_seq = x_pre_forecast.permute(0, 2, 1)   # [B, D, H]

        # 对「时间维 H」做线性映射到 d_model，得到变量 embedding
        h = self.pre_encoding(var_seq)              # [B, D, d_model]
        h = self.act(h)
        h = self.dropout(h)

        # ========= 4. 在变量维 D 上跑 LSTM（joint mixing over variates）=========
        # 正向 view
        out_fwd, _ = self.lstm(h)                   # [B, D, d_model]

        if self.backcast:
            # 反向 view：变量顺序翻转
            h_rev = torch.flip(h, dims=[1])         # [B, D, d_model]
            out_bwd, _ = self.lstm(h_rev)           # [B, D, d_model]
            out_bwd = torch.flip(out_bwd, dims=[1]) # 再翻回来

            h_cat = torch.cat([out_fwd, out_bwd], dim=-1)  # [B, D, 2*d_model]
        else:
            h_cat = out_fwd                                  # [B, D, d_model]

        h_cat = self.act(h_cat)
        h_cat = self.dropout(h_cat)

        # ========= 5. 映射回时间维 H =========
        # [B, D, in_dim_view] -> [B, D, H]
        x_out = self.fc(h_cat)                     # [B, D, H]

        # 再还原成 [B, H, D]
        x_out = x_out.permute(0, 2, 1)             # [B, H, D]

        # ========= 6. RevIN 反标准化 =========
        if self.revin:
            x_out = self.revin_layer(x_out, "denorm")

        # ========= 7. 输出 =========
        # shape: [B, pred_len, D]
        return x_out
