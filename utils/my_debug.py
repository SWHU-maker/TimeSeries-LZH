import matplotlib.pyplot as plt
import numpy as np
import torch 

def plot_and_save_pdf(data=None, save_path="./temp.pdf"):
    """
    生成图像并保存为PDF文件到指定路径

    参数:
        x: 可选，输入数据，如果未提供则生成默认示例数据
        save_path: 保存PDF的路径，默认是./temp.pdf
    """
    
    if isinstance(data, np.ndarray):
        data = data.astype(float)
    elif isinstance(data, t.Tensor):
        data = data.cpu().detach().numpy().astype(float)

        
    # 创建图像
    plt.figure(figsize=(8, 6))
    x = range(len(data))
    plt.plot(x, data, label="sin(x)", color="blue", linewidth=2)
    # 添加标题、标签和图例
    plt.title("Figure", fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    # 调整布局，防止标签被截断
    plt.tight_layout()
    # 保存为PDF文件
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    # 关闭图像释放内存
    plt.close()
    print(f"图像已成功保存到: {save_path}")
    return True


# 调用示例
if __name__ == "__main__":
    # 方式1：使用默认数据生成并保存
    plot_and_save_pdf(torch.randn(32, 96, 21)[0][:, 0])

    # 方式2：使用自定义数据
    # custom_x = np.linspace(0, 20, 200)
    # plot_and_save_pdf(custom_x)

    # 方式3：保存到自定义路径
    # plot_and_save_pdf(save_path="./my_plot.pdf")
