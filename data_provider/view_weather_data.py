import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# -------------------------- 核心配置（直接修改这里，和项目config对齐）--------------------------
# 数据配置
DATA_PATH = "./datasets/weather.csv"  # 数据集路径（和项目一致）
DATASET = "weather"                  # 数据集名称
# 时序参数
SEQ_LEN = 96                         # 输入序列长度（历史时间步）
PRED_LEN = 24                        # 预测序列长度（未来时间步）
# 数据划分
SPLITER_RATIO = "7:2:1"              # 训练/验证/测试比例
BATCH_SIZE = 32                      # DataLoader批次大小
# 其他配置
TS_VAR = 1                           # 参考项目get_ts：1=预测所有特征，0=预测最后一列
SHUFFLE_TRAIN = False                # 训练集是否打乱（查看数据时设为False，保持时序）
PRINT_ROWS = 3                       # 每个阶段打印前3行
# ----------------------------------------------------------------------------------

class MockConfig:
    """模拟项目的Config对象（无需导入get_config，直接在这定义参数）"""
    def __init__(self):
        self.dataset = DATASET
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.bs = BATCH_SIZE
        self.spliter_ratio = SPLITER_RATIO
        self.ts_var = TS_VAR
        self.shuffle = SHUFFLE_TRAIN
        self.use_train_size = False
        self.train_size = 0
        self.eval_set = True
        self.classification = False
        self.debug = False
        
        # 模拟日志（仅打印关键信息）
        class SimpleLog:
            def only_print(self, msg):
                print(f"📌 {msg}")
        self.log = SimpleLog()

def get_ts(data_path, config):
    """复现项目 get_ts 逻辑：读取数据+提取时间特征+归一化"""
    # 读取CSV数据
    df = pd.read_csv(data_path).to_numpy()
    if config.ts_var == 1:
        x, y = df[:, 1:], df[:, 1:]  # 所有特征作为输入和目标
    else:
        x, y = df[:, -1:], df[:, -1:]  # 仅最后一列作为输入和目标
    
    # 提取时间特征（月/日/周/时）
    timestamps = pd.to_datetime(df[:, 0])
    time_features = np.array([[ts.month, ts.day, ts.weekday(), ts.hour] for ts in timestamps])
    
    # 归一化（复现项目 get_scaler 逻辑，用MinMaxScaler 0-1缩放）
    def minmax_scaler(data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(data), scaler
    
    x_scaled, x_scaler = minmax_scaler(x)
    y_scaled, y_scaler = minmax_scaler(y)
    
    # 拼接时间特征和数值特征（x包含：4个时间特征 + N个天气特征）
    x_combined = np.concatenate((time_features, x_scaled), axis=1).astype(np.float32)
    y_scaled = y_scaled.astype(np.float32)
    
    return x_combined, y_scaled, x_scaler, y_scaler

def parse_split_ratio(ratio_str):
    """复现项目 parse_split_ratio 逻辑：解析7:2:1为比例"""
    parts = list(map(int, ratio_str.strip().split(':')))
    total = sum(parts)
    return [p / total for p in parts]

def get_train_valid_test_dataset(x, y, config):
    """复现项目数据划分逻辑：按比例划分，时序不打乱"""
    train_ratio, valid_ratio, _ = parse_split_ratio(config.spliter_ratio)
    
    # 计算划分大小
    if config.use_train_size:
        train_size = int(config.train_size)
    else:
        train_size = int(len(x) * train_ratio)
    
    valid_size = int(len(x) * valid_ratio) if config.eval_set else 0
    
    # 时序数据不打乱（保持顺序）
    train_x = x[:train_size]
    train_y = y[:train_size]
    valid_x = x[train_size:train_size + valid_size]
    valid_y = y[train_size:train_size + valid_size]
    test_x = x[train_size + valid_size:]
    test_y = y[train_size + valid_size:]
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y

class TimeSeriesDataset(Dataset):
    """复现项目 TimeSeriesDataset 逻辑：构建时序输入输出对"""
    def __init__(self, x, y, mode, config):
        self.x = x  # 形状：(时间步, 4+N特征)
        self.y = y  # 形状：(时间步, N特征)
        self.config = config
        self.mode = mode
    
    def __len__(self):
        """样本数 = 总时间步 - 输入长度 - 预测长度 + 1"""
        return len(self.x) - self.config.seq_len - self.config.pred_len + 1
    
    def __getitem__(self, idx):
        """滑动窗口取数据：x取天气特征，x_mark取时间特征"""
        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len
        
        # x：天气特征（去掉前4个时间特征）→ (seq_len, N特征)
        x = self.x[s_begin:s_end][:, 4:]
        # x_mark：时间特征（前4列：月/日/周/时）→ (seq_len, 4)
        x_mark = self.x[s_begin:s_end][:, :4]
        # y：目标值 → (pred_len, N特征)
        y = self.y[r_begin:r_end]
        
        return torch.tensor(x), torch.tensor(x_mark), torch.tensor(y)
    
    @staticmethod
    def custom_collate_fn(batch):
        """复现项目 collate_fn 逻辑：批量处理数据"""
        x, x_mark, y = zip(*batch)
        x = torch.stack(x)
        x_mark = torch.stack(x_mark)
        y = torch.stack(y)
        return x, x_mark, y

def get_dataloaders(train_set, valid_set, test_set, config):
    """复现项目 DataLoader 创建逻辑：适配系统设置多线程"""
    import platform
    import multiprocessing
    
    # 根据系统设置worker数（避免Windows报错）
    if platform.system() == 'Linux' and 'ubuntu' in platform.version().lower():
        max_workers = multiprocessing.cpu_count() // 5
        prefetch_factor = 2
    else:
        max_workers = 0
        prefetch_factor = None
    
    # 训练集DataLoader（可选打乱）
    train_loader = DataLoader(
        train_set,
        batch_size=config.bs,
        shuffle=config.shuffle,
        drop_last=False,
        pin_memory=True,
        collate_fn=TimeSeriesDataset.custom_collate_fn,
        num_workers=max_workers,
        prefetch_factor=prefetch_factor
    )
    
    # 验证集/测试集不打乱
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=TimeSeriesDataset.custom_collate_fn,
        num_workers=max_workers,
        prefetch_factor=prefetch_factor
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config.bs,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=TimeSeriesDataset.custom_collate_fn,
        num_workers=max_workers,
        prefetch_factor=prefetch_factor
    )
    
    return train_loader, valid_loader, test_loader

def print_raw_data_info(x, y):
    """打印原始数据（get_ts处理后）关键信息"""
    print("="*60)
    print("【1. 原始数据（预处理后）】")
    print("="*60)
    print(f"数据形状：x={x.shape} (时间步 × 特征数)，y={y.shape} (时间步 × 目标特征数)")
    print(f"特征构成：4个时间特征（月/日/周/时） + {x.shape[1]-4}个天气特征")
    print(f"总时间步：{x.shape[0]}")
    
    # 打印前3行（时间特征+天气特征前5列）
    time_cols = ["month", "day", "weekday", "hour"]
    weather_cols = [f"weather_feat_{i}" for i in range(5)]  # 只显示前5个天气特征
    x_display = pd.DataFrame(x[:PRINT_ROWS, :4+5], columns=time_cols + weather_cols)
    print(f"\n前{PRINT_ROWS}行数据（时间特征+前5个天气特征）：")
    print(x_display.round(4))
    print()

def print_split_info(train_x, train_y, valid_x, valid_y, test_x, test_y):
    """打印划分后数据集信息"""
    print("="*60)
    print("【2. 数据集划分结果】")
    print("="*60)
    print(f"训练集：{train_x.shape[0]}个时间步 → 样本数：{len(TimeSeriesDataset(train_x, train_y, 'train', config))}")
    print(f"验证集：{valid_x.shape[0]}个时间步 → 样本数：{len(TimeSeriesDataset(valid_x, valid_y, 'valid', config))}")
    print(f"测试集：{test_x.shape[0]}个时间步 → 样本数：{len(TimeSeriesDataset(test_x, test_y, 'test', config))}")
    print()

def print_dataloader_batch(loader, mode="训练集"):
    """打印DataLoader批次（核心：模型真实输入格式）"""
    print("="*60)
    print(f"【3. {mode} DataLoader 批次详情】")
    print("="*60)
    
    # 取第一个批次
    batch = next(iter(loader))
    x, x_mark, y = batch
    
    print(f"批次形状（batch_size={config.bs}）：")
    print(f"  - 天气特征 x：{x.shape} (批次大小 × 输入时间步 × 天气特征数)")
    print(f"  - 时间标记 x_mark：{x_mark.shape} (批次大小 × 输入时间步 × 时间特征数)")
    print(f"  - 预测目标 y：{y.shape} (批次大小 × 预测时间步 × 天气特征数)")
    
    # 打印第一个样本的前3个时间步
    print(f"\n第一个样本 - 时间标记 x_mark（前{PRINT_ROWS}步）：")
    x_mark_sample = x_mark[0][:PRINT_ROWS].numpy()
    print(pd.DataFrame(x_mark_sample, columns=["month", "day", "weekday", "hour"]).round(0))
    
    print(f"\n第一个样本 - 天气特征 x（前{PRINT_ROWS}步，前5个特征）：")
    x_sample = x[0][:PRINT_ROWS, :5].numpy()
    print(pd.DataFrame(x_sample, columns=[f"feat_{i}" for i in range(5)]).round(4))
    
    print(f"\n第一个样本 - 预测目标 y（前{PRINT_ROWS}步，前5个特征）：")
    y_sample = y[0][:PRINT_ROWS, :5].numpy()
    print(pd.DataFrame(y_sample, columns=[f"feat_{i}" for i in range(5)]).round(4))
    print()

def main():
    global config
    try:
        # 1. 初始化配置
        config = MockConfig()
        print(f"✅ 配置初始化完成：dataset={config.dataset}，seq_len={config.seq_len}，pred_len={config.pred_len}")
        print()
        
        # 2. 读取并预处理数据（复现get_ts逻辑）
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"❌ 数据集不存在：{DATA_PATH}，请检查路径是否正确")
        x, y, x_scaler, y_scaler = get_ts(DATA_PATH, config)
        print("✅ 数据预处理完成（读取+时间特征提取+归一化）")
        
        # 3. 划分数据集（复现项目划分逻辑）
        train_x, train_y, valid_x, valid_y, test_x, test_y = get_train_valid_test_dataset(x, y, config)
        
        # 4. 创建Dataset（复现TimeSeriesDataset）
        train_set = TimeSeriesDataset(train_x, train_y, 'train', config)
        valid_set = TimeSeriesDataset(valid_x, valid_y, 'valid', config)
        test_set = TimeSeriesDataset(test_x, test_y, 'test', config)
        
        # 5. 创建DataLoader（复现项目DataLoader逻辑）
        train_loader, valid_loader, test_loader = get_dataloaders(train_set, valid_set, test_set, config)
        config.log.only_print(f"DataLoader创建完成：训练集{len(train_loader)}批次，验证集{len(valid_loader)}批次，测试集{len(test_loader)}批次")
        print()
        
        # 6. 打印关键信息（精简版）
        print_raw_data_info(x, y)
        print_split_info(train_x, train_y, valid_x, valid_y, test_x, test_y)
        print_dataloader_batch(train_loader, mode="训练集")
        
        # 7. 核心总结
        print("="*60)
        print("【数据查看核心总结】")
        print("="*60)
        weather_feat_num = x.shape[1] - 4  # 天气特征数（总特征数-4个时间特征）
        print(f"1. 模型输入：天气特征({weather_feat_num}维) + 时间特征(4维)")
        print(f"2. 输入格式：(batch_size, {config.seq_len}, {weather_feat_num})")
        print(f"3. 输出格式：(batch_size, {config.pred_len}, {weather_feat_num})")
        print(f"4. 总样本数：训练集{len(train_set)} + 验证集{len(valid_set)} + 测试集{len(test_set)} = {len(train_set)+len(valid_set)+len(test_set)}")
        print("✅ 独立脚本运行完成（无任何项目组件依赖）")
    
    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")

if __name__ == "__main__":
    main()
