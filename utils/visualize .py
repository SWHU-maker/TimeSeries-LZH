# coding : utf-8
# Author : Yuxiang Zeng
# utils/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_and_save(data=None, save_path="./temp.pdf", plot_type='line', xlabel=None, ylabel=None, title=None, cmap='viridis'):
    """
    生成图像并保存为文件

    参数:
        data: 可选，输入数据，如果未提供则生成默认示例数据
        save_path: 保存图像的路径，默认为 ./temp.pdf
        plot_type: 'line', 'heatmap', 'distribution'，选择图像类型
        xlabel, ylabel: x、y轴标签
        title: 图表标题
        cmap: 热力图的颜色映射
    """
    if data is None:
        # 如果没有提供数据，则使用默认的示例数据
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

    # 将输入数据转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy().astype(float)
    elif isinstance(data, np.ndarray):
        data = data.astype(float)

    # 创建图像
    plt.figure(figsize=(8, 6))

    if plot_type == 'line':
        # 线图
        x = range(len(data))
        plt.plot(x, data, label="Data", color="blue", linewidth=2)
        plt.xlabel(xlabel if xlabel else "X", fontsize=12)
        plt.ylabel(ylabel if ylabel else "Y", fontsize=12)
    elif plot_type == 'heatmap':
        # 热力图
        plt.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar()
        plt.xlabel(xlabel if xlabel else "Time Step", fontsize=12)
        plt.ylabel(ylabel if ylabel else "Feature", fontsize=12)
    elif plot_type == 'distribution':
        # 分布图（例如直方图）
        plt.hist(data.flatten(), bins=50)
        plt.xlabel(xlabel if xlabel else "Value", fontsize=12)
        plt.ylabel(ylabel if ylabel else "Frequency", fontsize=12)

    if title:
        plt.title(title, fontsize=14)

    # 添加标题、标签和图例
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存为指定路径
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已成功保存到: {save_path}")
    return True
