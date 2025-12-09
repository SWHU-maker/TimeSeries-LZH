import os
import argparse
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

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
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DiffusionLZHConfig:
    cfg = DiffusionLZHConfig()
    for field in ["data_path", "save_dir", "save_name", "epochs", "lr", "seq_len", "batch_size", "noise_steps", "device"]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    return cfg


def split_dataset(dataset, cfg: DiffusionLZHConfig):
    n_total = len(dataset)
    n_train = int(n_total * cfg.train_split)
    n_val = int(n_total * cfg.val_split)
    n_test = n_total - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test])


def train_epoch(model: DiffusionWrapper, loader, optimizer, scaler, cfg: DiffusionLZHConfig):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss, _ = model(batch)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if step % cfg.log_interval == 0:
            print(f"[train] step {step}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model: DiffusionWrapper, loader):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        loss, _ = model(batch)
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def save_checkpoint(model: DiffusionWrapper, cfg: DiffusionLZHConfig, feature_dim: int, path: str, d_model: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = model.export_state()
    payload.update(
        {
            "feature_dim": feature_dim,
            "d_model": d_model,
            "config": asdict(cfg),
        }
    )
    torch.save(payload, path)
    print(f"[save] checkpoint -> {path}")


def main():
    args = parse_args()
    cfg = build_config(args)

    dataset = WeatherSequenceDataset(csv_path=cfg.data_path, seq_len=cfg.seq_len, feature_cols=cfg.feature_cols)
    feature_dim = len(dataset.feature_cols)
    train_ds, val_ds, test_ds = split_dataset(dataset, cfg)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, drop_last=True)

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    model = DiffusionWrapper(input_size=feature_dim, seq_len=cfg.seq_len, d_model=args.d_model, cfg=cfg)
    device = model.device
    print(f"[info] device={device}, feature_dim={feature_dim}, seq_len={cfg.seq_len}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch {epoch}/{cfg.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scaler, cfg)
        val_loss = eval_epoch(model, val_loader)
        print(f"[summary] train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            # save_path = os.path.join(cfg.save_dir, cfg.save_name)
            save_path = os.path.join(cfg.save_dir, f"{cfg.save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            save_checkpoint(model, cfg, feature_dim, save_path, args.d_model)

    # 最终在测试集简单汇报
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    test_loss = eval_epoch(model, test_loader)
    print(f"[test] loss={test_loss:.4f}")


if __name__ == "__main__":
    main()

