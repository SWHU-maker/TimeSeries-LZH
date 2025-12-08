# coding : utf-8
# Author : Yuxiang Zeng
# utils/visualize.py

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_1d(data):
    """把各种类型数据安全地转成 1D numpy 向量，用于画线。"""
    # Tensor -> numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        data = data
    else:
        # 列表 / 可迭代等
        data = np.array(data)

    data = data.astype(float)

    # 如果是多维的，默认拉平成一条线
    if data.ndim > 1:
        data = data.reshape(-1)

    return data


def plot_and_save(
    data=None,
    save_path: str = "./debug_visualizations/temp.pdf",
    xlabel: str = "Index",
    ylabel: str = "Value",
    title: str = "Line Plot",
    **kwargs,  # 兼容之前多余的参数，比如 plot_type, cmap 等，直接忽略
):
    """
    只画【折线图】并保存到文件（自动创建目录）

    参数:
        data: 输入数据，可以是 Tensor / ndarray / list，支持多维，内部会拉平成一条线
        save_path: 保存图像的路径，例如 "./debug_visualizations/x_before_revIN.pdf"
        xlabel, ylabel: 坐标轴标签
        title: 图表标题
    """
    # 1. 准备数据
    if data is None:
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

    y = _to_numpy_1d(data)
    x = np.arange(len(y))

    # 2. 自动创建保存目录
    dir_path = os.path.dirname(save_path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)

    # 3. 画图（折线图）
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=1.8)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 4. 根据后缀名自动选择格式（默认 pdf）
    ext = os.path.splitext(save_path)[1].lower()
    if ext.startswith("."):
        ext = ext[1:]
    if ext == "":
        ext = "pdf"

    plt.savefig(save_path, format=ext, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_and_save] 图像已成功保存到: {save_path}")
    return True
