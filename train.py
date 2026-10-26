from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from losses import BCEDiceLoss
from metrics import merge_metric_batches, segmentation_metrics
from data import SegmentationDataset, load_manifest, make_fold, save_records, seed_everything
from model import MODEL_FLAGS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net++ segmentation experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--model", choices=list(MODEL_FLAGS), default="aspp_cbam_se")
    parser.add_argument("--fold", default="0", help="0-4 or all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool, amp: bool) -> tuple[float, dict[str, float]]:
    model.train(train)
    losses, metric_items, metric_weights = [], [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(float(loss.detach().cpu()))
            metric_items.append(segmentation_metrics(logits, masks))
            metric_weights.append(int(images.shape[0]))
    return sum(loss * n for loss, n in zip(losses, metric_weights)) / max(1, sum(metric_weights)), merge_metric_batches(metric_items, metric_weights)


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_fold(args: argparse.Namespace, records, fold: int, device: torch.device) -> dict[str, object]:
    fold_dir = Path(args.output) / f"fold_{fold}"
    if (fold_dir / "best.pt").exists():
        raise FileExistsError(f"Existing run: {fold_dir}. Choose a new --output directory.")
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed + fold)
    train_records, val_records, test_records = make_fold(records, fold, args.folds, args.seed)
    save_records(fold_dir / "train.csv", train_records)
    save_records(fold_dir / "val.csv", val_records)
    save_records(fold_dir / "test.csv", test_records)
    train_ds = SegmentationDataset(train_records, args.image_size, augment=True)
    val_ds = SegmentationDataset(val_records, args.image_size, augment=False)
    test_ds = SegmentationDataset(test_records, args.image_size, augment=False)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    model = build_model(args.model, in_channels=3, num_classes=1, base_channels=args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    amp = not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history, best_dice, best_epoch, stale = [], -1.0, -1, 0
    best_path = fold_dir / "best.pt"
    started = time.time()
    print(f"Fold {fold}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} device={device} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True, amp)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False, amp)
        scheduler.step(val_metrics["dice"])
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss, "val_loss": val_loss}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f}")
        if val_metrics["dice"] > best_dice:
            best_dice, best_epoch, stale = val_metrics["dice"], epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "fold": fold, "best_epoch": epoch, "best_val_dice": best_dice}, best_path)
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}; best epoch={best_epoch}")
            break
    write_history(fold_dir / "history.csv", history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, optimizer, scaler, device, False, amp)
    result = {"fold": fold, "best_epoch": best_epoch, "best_val_dice": best_dice, "test_loss": test_loss, **{f"test_{k}": v for k, v in test_metrics.items()}, "seconds": time.time() - started}
    (fold_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.image_size < 32 or args.base_channels < 1:
        raise ValueError("epochs/patience/batch-size/base-channels must be positive; image-size >= 32")
    device = select_device(args.device)
    records = load_manifest(args.manifest)
    folds = list(range(args.folds)) if args.fold == "all" else [int(args.fold)]
    if any((Path(args.output) / f"fold_{fold}" / "best.pt").exists() for fold in folds):
        raise FileExistsError("Output contains trained folds; choose a new --output.")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    results = [train_fold(args, records, fold, device) for fold in folds]
    (Path(args.output) / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
