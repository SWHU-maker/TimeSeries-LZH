# coding : utf-8
# Author : Yuxiang Zeng
# TimeMixer.py
# TimeMixer baseline for weather forecasting

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block using DFT
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList([
            nn.Sequential(
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** i),
                ),
                nn.GELU(),
                torch.nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** i),
                ),
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, config):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.down_sampling_window = config.down_sampling_window
        self.down_sampling_layers = config.down_sampling_layers
        self.decomp_method = config.decomp_method
        self.dropout = config.dropout

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

        # Decomposition
        if self.decomp_method == 'moving_avg':
            self.decomposition = series_decomp(config.moving_avg)
        elif self.decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(config.top_k)
        else:
            raise ValueError('decomposition method error')

        # Cross layer for channel mixing
        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=self.d_model),
        )

        # Mixing season and trend
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            self.seq_len, self.down_sampling_window, self.down_sampling_layers
        )
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            self.seq_len, self.down_sampling_window, self.down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=self.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            season = self.cross_layer(season)
            trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class TimeMixer(nn.Module):
    """
    TimeMixer for weather forecasting
    Adapted to LZH's framework
    """
    def __init__(self, input_size, config):
        super(TimeMixer, self).__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.d_model = config.d_model
        self.input_size = input_size
        self.e_layers = config.e_layers
        self.down_sampling_window = config.down_sampling_window
        self.down_sampling_layers = config.down_sampling_layers
        self.down_sampling_method = config.down_sampling_method
     

        # Embedding
        self.value_embedding = nn.Linear(input_size, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # PDM blocks
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(config) for _ in range(self.e_layers)
        ])

        # Normalization layers for multi-scale
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(self.input_size, affine=True)
            for i in range(self.down_sampling_layers + 1)
        ])

        # Prediction layers for multi-scale
        self.predict_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                self.seq_len // (self.down_sampling_window ** i),
                self.pred_len,
            )
            for i in range(self.down_sampling_layers + 1)
        ])

        # Final projection
        self.projection_layer = nn.Linear(self.d_model, input_size, bias=True)

    def __multi_scale_process_inputs(self, x_enc):
        """
        Multi-scale down-sampling
        """
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            down_pool = nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.input_size,
                kernel_size=3,
                padding=1,
                stride=self.down_sampling_window,
                padding_mode='circular',
                bias=False
            )
        else:
            return [x_enc]

        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def forward(self, x, x_mark):
        """
        x: [B, seq_len, input_size]
        x_mark: [B, seq_len, 4] (time features, not used in TimeMixer)
        """
        # Multi-scale processing
        x_enc = self.__multi_scale_process_inputs(x)

        # Normalize and embed
        x_list = []
        for i, x_scale in enumerate(x_enc):
            B, T, N = x_scale.size()
            # Normalize
            x_scale = self.normalize_layers[i](x_scale, 'norm')
            # Embed
            x_scale = self.value_embedding(x_scale)
            x_scale = self.dropout(x_scale)
            x_list.append(x_scale)

        # Past Decomposable Mixing
        for i in range(self.e_layers):
            x_list = self.pdm_blocks[i](x_list)

        # Future prediction for each scale
        dec_out_list = []
        for i, enc_out in enumerate(x_list):
            # Temporal dimension alignment
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            # Project to output dimension
            dec_out = self.projection_layer(dec_out)
            dec_out_list.append(dec_out)

        # Sum all scales
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        
        # Denormalize
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out