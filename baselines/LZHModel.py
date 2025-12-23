# LZHModel.py

# coding : utf-8
# Author : zhenghao Luo
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
from utils.delete import delete_small_log_files
from utils.visualize import  plot_and_save



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


class LZHModel(nn.Module):
    def __init__(self, input_size, d_model, revin, num_heads, num_layers, seq_len, pred_len, use_diff: int = 0, noise_scale=1, noise_steps=100, diffusion_ckpt: str = None):
        super().__init__()

        self.revin = revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_diff = bool(use_diff)  # use_diff=0: 不创建也不使用, use_diff=1: 创建并使用

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
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.use_pretrained_diffusion = False  # 标记是否使用了预训练扩散模型
        self._diffusion_forward_called = False  # 标记是否已经调用过 diffusion_forward
        
        # use_diff=0: 不创建也不使用
        # use_diff=1: 创建并使用（如果有 diffusion_ckpt 则加载预训练并冻结，否则初始化跟随训练）
        if self.use_diff:
            if diffusion_ckpt:
                self._load_diffusion_from_ckpt(diffusion_ckpt, input_size, seq_len, d_model, noise_scale, noise_steps)
                self.use_pretrained_diffusion = True
            else:
                self._init_diffusion(input_size, seq_len, d_model, noise_scale, noise_steps)
                print(f"[INFO] 扩散模型已初始化，将跟随训练 | noise_scale={noise_scale}, noise_steps={noise_steps}")
        else:
            self.diffusion = None
            self.reverse = None
            print("[INFO] 扩散模型未启用 (use_diff=0)")
        
        # 打印模型配置摘要
        print("=" * 80)
        print("[LZHModel 配置摘要]")
        print(f"  use_diff: {self.use_diff}")
        print(f"  use_pretrained_diffusion: {self.use_pretrained_diffusion}")
        if self.use_diff:
            print(f"  扩散模型状态: {'预训练模型（已冻结）' if self.use_pretrained_diffusion else '新模型（跟随训练）'}")
            if diffusion_ckpt:
                print(f"  预训练路径: {diffusion_ckpt}")
        print(f"  device: {self.device}")
        print(f"  input_size: {input_size}, seq_len: {seq_len}, pred_len: {pred_len}")
        print(f"  d_model: {d_model}, num_layers: {num_layers}, num_heads: {num_heads}")
        print("=" * 80)

    def _init_diffusion(self, input_size, seq_len, d_model, noise_scale, noise_steps):
        self.diffusion = gd.GaussianDiffusion(
            mean_type=gd.ModelMeanType.EPSILON,
            noise_schedule='linear-var',
            noise_scale=noise_scale,
            noise_min=0.0001,
            noise_max=0.02,
            steps=noise_steps,
            device=self.device
        )
        self.reverse = DNN(in_dims=[input_size * seq_len, input_size * seq_len],
                           out_dims=[input_size * seq_len, input_size * seq_len],
                           emb_size=d_model).to(self.device)

    def _load_diffusion_from_ckpt(self, ckpt_path, input_size, seq_len, d_model, default_noise_scale, default_noise_steps):
        state = torch.load(ckpt_path, map_location=self.device)
        params = state.get("diffusion_params", {})
        noise_scale = params.get("noise_scale", default_noise_scale)
        noise_min = params.get("noise_min", 0.0001)
        noise_max = params.get("noise_max", 0.02)
        steps = params.get("steps", default_noise_steps)
        schedule = params.get("noise_schedule", "linear-var")
        mean_type = params.get("mean_type", "EPSILON")
        self.diffusion = gd.GaussianDiffusion(
            mean_type=getattr(gd.ModelMeanType, mean_type),
            noise_schedule=schedule,
            noise_scale=noise_scale,
            noise_min=noise_min,
            noise_max=noise_max,
            steps=steps,
            device=self.device
        )
        self.reverse = DNN(in_dims=[input_size * seq_len, input_size * seq_len],
                           out_dims=[input_size * seq_len, input_size * seq_len],
                           emb_size=state.get("d_model", d_model)).to(self.device)
        if "reverse_state_dict" in state:
            self.reverse.load_state_dict(state["reverse_state_dict"])
            # 冻结预训练扩散模型的参数，不继续训练
            for param in self.reverse.parameters():
                param.requires_grad = False
            self.reverse.eval()  # 设置为评估模式
            print(
                f"[INFO] Loaded pretrained diffusion ckpt: {ckpt_path} | "
                f"mean_type={mean_type}, schedule={schedule}, steps={steps}, "
                f"noise_scale={noise_scale}, noise_min={noise_min}, noise_max={noise_max} | "
                f"扩散模型参数已冻结，仅用于推理"
            )
        else:
            raise ValueError(f"未在 {ckpt_path} 找到 reverse_state_dict")

    def diffusion_forward(self, y):
        """执行扩散模型去噪/训练损失计算。"""
        # use_diff=0: 直接返回原输入，不进行任何处理
        if not self.use_diff:
            self.diffusion_loss = 0.0
            return y

        # use_diff=1: 使用扩散模型进行去噪（此时 self.diffusion 一定不为 None）
        raw_shape = y.shape
        y_flat = y.reshape(y.shape[0], -1) 
        
        # 首次调用时打印状态信息
        if not self._diffusion_forward_called:
            self._diffusion_forward_called = True
            print("[INFO] diffusion_forward 首次调用")
            print(f"  训练模式 (self.training): {self.training}")
            print(f"  使用预训练扩散模型 (use_pretrained_diffusion): {self.use_pretrained_diffusion}")
            print(f"  输入形状: {y.shape}")
        
        # 如果使用了预训练模型，直接进行去噪，不计算训练损失
        if self.use_pretrained_diffusion:
            with torch.no_grad():
                y_denoised = self.diffusion.p_sample(self.reverse, y_flat, 60, False)
            self.diffusion_loss = 0.0
            if not hasattr(self, '_print_pretrained_info'):
                print(f"  [扩散模型] 使用预训练模型进行去噪（参数已冻结，仅推理）")
                self._print_pretrained_info = True
        elif self.training:
            # 未使用预训练模型时，正常计算训练损失
            diff_output = self.diffusion.training_losses(self.reverse, y_flat, True)
            y_denoised = diff_output["pred_xstart"]
            self.diffusion_loss = diff_output["loss"].mean()
            if not hasattr(self, '_print_training_info'):
                print(f"  [扩散模型] 训练模式：计算扩散损失（新模型跟随训练）")
                self._print_training_info = True
        else:
            # 推理模式
            y_denoised = self.diffusion.p_sample(self.reverse, y_flat, 5, False)
            self.diffusion_loss = 0.0
            if not hasattr(self, '_print_eval_info'):
                print(f"  [扩散模型] 推理模式：使用扩散模型进行去噪（新模型）")
                self._print_eval_info = True
            
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
        # from utils.visualize import plot_and_save
        # plot_and_save(x[0, :, 0], save_path="./debug_visualizations/preDiffusion/weather_before_preDiff_ch0.pdf", title="before preDiffusion | sample0 ch0")
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
