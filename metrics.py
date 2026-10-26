from __future__ import annotations

import numpy as np
import torch


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy().astype(bool)
    targets_np = targets.detach().cpu().numpy().astype(bool)
    values = {key: [] for key in ("dice", "iou", "precision", "recall")}
    for pred, target in zip(preds, targets_np):
        pred, target = pred[0], target[0]
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        pred_sum, target_sum = pred.sum(), target.sum()
        tp = intersection
        fp = np.logical_and(pred, ~target).sum()
        fn = np.logical_and(~pred, target).sum()
        values["dice"].append((2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7))
        values["iou"].append((intersection + 1e-7) / (union + 1e-7))
        values["precision"].append((tp + 1e-7) / (tp + fp + 1e-7))
        values["recall"].append((tp + 1e-7) / (tp + fn + 1e-7))
    return {key: float(np.mean(value)) for key, value in values.items()}


def merge_metric_batches(items: list[dict[str, float]], weights: list[int] | None = None) -> dict[str, float]:
    if not items:
        return {key: 0.0 for key in ("dice", "iou", "precision", "recall")}
    if weights is None:
        weights = [1] * len(items)
    total = float(sum(weights))
    return {key: float(sum(item[key] * weight for item, weight in zip(items, weights)) / total) for key in items[0]}
