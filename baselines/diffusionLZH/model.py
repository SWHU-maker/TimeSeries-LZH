import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Optional
import layers.diffusion.gaussian_diffusion as gd
from layers.diffusion.DNN import DNN
from .config import DiffusionLZHConfig


class DiffusionWrapper(nn.Module):
    """
    封装 GaussianDiffusion + 逆向网络 DNN，提供训练与采样接口。
    """

    def __init__(self, input_size: int, seq_len: int, d_model: int, cfg: DiffusionLZHConfig):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.cfg = cfg
        self.device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.diffusion = gd.GaussianDiffusion(
            mean_type=getattr(gd.ModelMeanType, cfg.mean_type),
            noise_schedule=cfg.noise_schedule,
            noise_scale=cfg.noise_scale,
            noise_min=cfg.noise_min,
            noise_max=cfg.noise_max,
            steps=cfg.noise_steps,
            device=self.device,
        )
        flat_dim = input_size * seq_len
        self.reverse = DNN(
            in_dims=[flat_dim, flat_dim],
            out_dims=[flat_dim, flat_dim],
            emb_size=d_model,
        ).to(self.device)
        self.d_model = d_model

    def forward(self, x):
        """
        x: (B, L, C) -> flatten -> diffusion 训练
        """
        b, l, c = x.shape
        assert l == self.seq_len and c == self.input_size, "输入尺寸与配置不一致"
        x_flat = x.reshape(b, -1).to(self.device)
        out = self.diffusion.training_losses(self.reverse, x_flat, reweight=True)
        loss = out["loss"].mean()
        return loss, out

    @torch.no_grad()
    def sample(self, x_start, steps: Optional[int] = None):
        """
        从给定 x_start（通常为干净数据）生成采样结果，主要用于调试。
        """
        steps = steps or self.cfg.noise_steps
        x_flat = x_start.reshape(x_start.shape[0], -1).to(self.device)
        out = self.diffusion.p_sample(self.reverse, x_flat, steps, sampling_noise=False)
        return out.reshape(x_start.shape)

    def export_state(self):
        return {
            "reverse_state_dict": self.reverse.state_dict(),
            "diffusion_params": {
                "mean_type": self.cfg.mean_type,
                "noise_schedule": self.cfg.noise_schedule,
                "noise_scale": self.cfg.noise_scale,
                "noise_min": self.cfg.noise_min,
                "noise_max": self.cfg.noise_max,
                "steps": self.cfg.noise_steps,
            },
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "config": asdict(self.cfg),
        }

