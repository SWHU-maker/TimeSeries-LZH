# LZHModel.py

# coding : utf-8
# Author : yuxiang Zeng (adapted for LZHModel by AI)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入使用的基础组件
from layers.att.self_attention import Attention 
from layers.feedforward.ffn import FeedForward 
from layers.revin import RevIN
from einops import rearrange
import layers.diffusion.gaussian_diffusion as gd
from layers.diffusion.DNN import DNN


# --- DataEmbedding 和子模块 ---

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                         kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        month_x = self.month_embed(x[:, :, 0])
        day_x = self.day_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        hour_x = self.hour_embed(x[:, :, 3])
        return month_x + day_x + weekday_x + hour_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x_out = self.value_embedding(x) + self.position_embedding(x)
        else:
            x_out = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x_out)

# 🚀 LZHModel
class LZHModel(nn.Module):
    def __init__(self, input_size, d_model, revin, num_heads, num_layers, seq_len, pred_len, diffusion=False, noise_scale=1, noise_steps=100):
        super().__init__()
        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        if self.revin:
            self.revin_layer = RevIN(num_features=input_size, affine=False, subtract_last=False)
            
        self.enc_embedding = DataEmbedding(input_size, d_model)
        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(d_model),
                        Attention(d_model, num_heads, dropout=0.10),
                        nn.LayerNorm(d_model),
                        FeedForward(d_model, d_ff=d_model * 2, dropout=0.10)
                    ]
                )
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, input_size)
        self.diffusion_loss = 0
        
        if diffusion:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.diffusion = gd.GaussianDiffusion(
                mean_type=gd.ModelMeanType.EPSILON,
                noise_schedule='linear-var',
                noise_scale=noise_scale,
                noise_min=0.0001,
                noise_max=0.02,
                steps=noise_steps,
                device=device
            )
            self.reverse = DNN(in_dims=[input_size * seq_len, input_size * seq_len], 
                               out_dims=[input_size * seq_len, input_size * seq_len], 
                               emb_size=d_model).to(device)
        else:
            self.diffusion = None
            self.reverse = None

    def diffusion_forward(self, y):
        """执行扩散模型去噪/训练损失计算。"""
        if self.diffusion is not None:
            raw_shape = y.shape
            y_flat = y.reshape(y.shape[0], -1) 
            
            if self.training:
                diff_output = self.diffusion.training_losses(self.reverse, y_flat, True)
                y_denoised = diff_output["pred_xstart"]
                self.diffusion_loss = diff_output["loss"].mean()
            else:
                y_denoised = self.diffusion.p_sample(self.reverse, y_flat, 5, False)
                self.diffusion_loss = 0.0
                
            y = y_denoised.reshape(raw_shape)
        return y
    
    def forward(self, x, x_mark=None, timesteps=None):
        # 1. RevIN 归一化
        if self.revin:
            # Note: 这里的实现与 Transformer2.py 中的实现相同，使用的是手动 RevIN 逻辑
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            
        # 2. 扩散模型去噪/增强
        x = self.diffusion_forward(x)
            
        # 3. 嵌入层
        x = self.enc_embedding(x, x_mark) 
        
        # 4. 序列长度投影 (Extrapolation)
        x = rearrange(x, 'bs seq_len d_model -> bs d_model seq_len')
        x = self.predict_linear(x)
        x = rearrange(x, 'bs d_model seq_pred_len -> bs seq_pred_len d_model')
        
        # 5. Transformer 编码器层
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        
        # 6. 最终归一化和投影
        x = self.norm(x)
        y = self.projection(x)[:, -self.pred_len:, :]
        
        # 7. RevIN 反归一化
        if self.revin:
            y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            
        return y

    def apply_noise(self, user_emb, item_emb, diff_model):
        raise NotImplementedError("This method is for diffusion training examples and is not used here.")

    # ====== 新增：保存 x / x_hat / y / y_pred 的可视化图像（全量样本） ======
    @torch.no_grad()
    def save_diffusion_plots(self, x, x_hat, y, y_pred,
                             save_dir: str = "show_diffusion",
                             prefix: str = "sample"):
        """
        将 x, x_hat, y, y_pred 画在一张图上，并为所有样本各保存一张图。
        只画最后一个通道（channel 维度的 -1）。

        参数：
            x      : [N, seq_len, C]     历史输入
            x_hat  : [N, seq_len, C]     历史输入经扩散/去噪后的结果
            y      : [N, pred_len, C]    真实未来
            y_pred : [N, pred_len, C]    预测未来
            save_dir : 保存图片的文件夹名（默认 "show_diffusion"）
            prefix   : 文件名前缀（默认 "sample"）
        """
        import os
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        # 确保在 CPU 上并 detach
        x = x.detach().cpu()
        x_hat = x_hat.detach().cpu()
        y = y.detach().cpu()
        y_pred = y_pred.detach().cpu()

        N, seq_len, C = x.shape
        Ny, pred_len, Cy = y.shape
        assert N == x_hat.shape[0] == y_pred.shape[0], "batch size 不一致"
        assert C == x_hat.shape[2] == Cy == y_pred.shape[2], "通道数不一致"

        ch = C - 1  # 最后一个通道

        for i in range(N):
            x_i = x[i, :, ch].numpy()
            xhat_i = x_hat[i, :, ch].numpy()
            y_i = y[i, :, ch].numpy()
            ypred_i = y_pred[i, :, ch].numpy()

            t_hist = list(range(seq_len))
            t_future = list(range(seq_len, seq_len + pred_len))

            plt.figure(figsize=(10, 4))
            plt.plot(t_hist, x_i, label="x")
            plt.plot(t_hist, xhat_i, label="x_hat")
            plt.plot(t_future, y_i, label="y")
            plt.plot(t_future, ypred_i, label="y_pred")

            # 历史/未来的分割线
            plt.axvline(seq_len - 1, linestyle=":", linewidth=1.0)

            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()

            fname = f"{prefix}_{i}.png"
            fpath = os.path.join(save_dir, fname)
            plt.savefig(fpath)
            plt.close()
