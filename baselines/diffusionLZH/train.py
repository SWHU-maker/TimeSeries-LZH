import os
import argparse
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from .config import DiffusionLZHConfig
from .dataset import WeatherSequenceDataset
from .model import DiffusionWrapper
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train standalone diffusion for LZHModel (weather dataset).")
    parser.add_argument("--data_path", type=str, default=None, help="路径，默认为 config 中的 data_path")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--noise_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DiffusionLZHConfig:
    cfg = DiffusionLZHConfig()
    for field in ["data_path", "save_dir", "save_name", "epochs", "lr", "seq_len", "batch_size", "noise_steps", "device", "patience"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    return cfg


def split_dataset_indices(n_total: int, seq_len: int, cfg: DiffusionLZHConfig):
    """
    修正：时间序列按连续时间段划分，避免滑窗重叠导致的数据泄露
    """
    # 计算原始时间序列的长度（考虑滑窗）
    n_timepoints = n_total + seq_len - 1
    
    # 按时间顺序划分时间点
    n_train_time = int(n_timepoints * cfg.train_split)
    n_val_time = int(n_timepoints * cfg.val_split)
    
    # 转换为样本索引（每个样本的起始位置）
    # 训练集：[0, n_train_time - seq_len]
    train_end = max(0, n_train_time - seq_len + 1)
    # 验证集：[n_train_time - seq_len + 1, n_train_time + n_val_time - seq_len]
    val_start = train_end
    val_end = max(val_start, n_train_time + n_val_time - seq_len + 1)
    # 测试集：[n_train_time + n_val_time - seq_len + 1, n_total]
    test_start = val_end
    
    train_indices = list(range(0, train_end))
    val_indices = list(range(val_start, val_end))
    test_indices = list(range(test_start, n_total))
    
    print(f"[split] 时间序列连续划分（避免重叠）:")
    print(f"  原始时间点数: {n_timepoints}")
    print(f"  训练时间段: [0, {n_train_time}), 样本数: {len(train_indices)}")
    print(f"  验证时间段: [{n_train_time}, {n_train_time + n_val_time}), 样本数: {len(val_indices)}")
    print(f"  测试时间段: [{n_train_time + n_val_time}, {n_timepoints}), 样本数: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def train_epoch(model: DiffusionWrapper, loader, optimizer, scaler, cfg: DiffusionLZHConfig):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss, _ = model(batch)
        
        # 检查NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[warning] NaN/Inf loss at step {step}, skipping...")
            continue
            
        scaler.scale(loss).backward()
        
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if step % cfg.log_interval == 0:
                print(f"[train] step {step}/{len(loader)} loss={loss.item():.6f} grad_norm={grad_norm:.4f}")
        else:
            if step % cfg.log_interval == 0:
                print(f"[train] step {step}/{len(loader)} loss={loss.item():.6f}")
        
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model: DiffusionWrapper, loader):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        loss, _ = model(batch)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)


def save_checkpoint(model: DiffusionWrapper, cfg: DiffusionLZHConfig, feature_dim: int, path: str, d_model: int, mean: np.ndarray, std: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = model.export_state()
    payload.update(
        {
            "feature_dim": feature_dim,
            "d_model": d_model,
            "config": asdict(cfg),
            "normalization_mean": mean,
            "normalization_std": std,
        }
    )
    torch.save(payload, path)
    print(f"[save] checkpoint -> {path}")


def main():
    args = parse_args()
    cfg = build_config(args)

    print("="*60)
    print("Training Configuration:")
    print(f"  Data path: {cfg.data_path}")
    print(f"  Sequence length: {cfg.seq_len}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Patience: {cfg.patience}")
    min_epochs = getattr(cfg, 'min_epochs', 10)
    print(f"  Min epochs: {min_epochs}")
    print("="*60)

    # 先创建完整数据集（不归一化）来获取索引
    full_dataset = WeatherSequenceDataset(csv_path=cfg.data_path, seq_len=cfg.seq_len, feature_cols=cfg.feature_cols)
    feature_dim = len(full_dataset.feature_cols)
    
    # 修正：按时间序列连续划分，避免滑窗重叠
    train_indices, val_indices, test_indices = split_dataset_indices(len(full_dataset), cfg.seq_len, cfg)
    
    # 只用训练集数据计算归一化参数
    train_data_list = []
    for idx in train_indices:
        if idx + cfg.seq_len <= len(full_dataset.data):
            train_data_list.append(full_dataset.data[idx:idx+cfg.seq_len])
    train_data = np.concatenate(train_data_list, axis=0)
    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True) + 1e-5
    
    print(f"\n[normalization] Using ONLY training data statistics:")
    print(f"  Mean: {train_mean.mean():.6f}, Std: {train_std.mean():.6f}")
    
    # 用训练集的归一化参数创建所有数据集
    train_dataset = WeatherSequenceDataset(cfg.data_path, cfg.seq_len, cfg.feature_cols, train_mean, train_std)
    val_dataset = WeatherSequenceDataset(cfg.data_path, cfg.seq_len, cfg.feature_cols, train_mean, train_std)
    test_dataset = WeatherSequenceDataset(cfg.data_path, cfg.seq_len, cfg.feature_cols, train_mean, train_std)
    
    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    test_ds = Subset(test_dataset, test_indices)
    
    # 数据诊断
    print("\n" + "="*60)
    print("Data Diagnostics:")
    sample = train_ds[0]
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample mean: {sample.mean().item():.6f}, std: {sample.std().item():.6f}")
    print(f"  Sample min: {sample.min().item():.6f}, max: {sample.max().item():.6f}")
    print("="*60 + "\n")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, drop_last=True)

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    model = DiffusionWrapper(input_size=feature_dim, seq_len=cfg.seq_len, d_model=args.d_model, cfg=cfg)
    device = model.device
    print(f"\n[info] Training on device: {device}")
    print(f"[info] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.epochs}")
        print('='*60)
        
        train_loss = train_epoch(model, train_loader, optimizer, scaler, cfg)
        val_loss = eval_epoch(model, val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[summary] Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Best Val:   {best_val:.6f} (Epoch {best_epoch})")
        print(f"  LR:         {current_lr:.8f}")
        
        improvement = (best_val - val_loss) / best_val if best_val != float("inf") else 0
        
        if val_loss < best_val * 0.999:
            best_val = val_loss
            best_epoch = epoch
            save_path = os.path.join(cfg.save_dir, cfg.save_name)
            save_checkpoint(model, cfg, feature_dim, save_path, args.d_model, train_mean, train_std)
            patience_counter = 0
            print(f"  ✓ New best model! Improvement: {improvement*100:.3f}%")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement. Patience: {patience_counter}/{cfg.patience}")
        
        if epoch >= min_epochs and patience_counter >= cfg.patience:
            print(f"\n[info] Early stopping triggered at epoch {epoch}.")
            print(f"[info] Best validation loss: {best_val:.6f} at epoch {best_epoch}")
            break

    # 测试集评估
    print("\n" + "="*60)
    print("Final Evaluation on Test Set...")
    print("="*60)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    test_loss = eval_epoch(model, test_loader)
    print(f"\n[test] Test Loss: {test_loss:.6f}")
    print(f"[test] Best Val Loss: {best_val:.6f} (Epoch {best_epoch})")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()