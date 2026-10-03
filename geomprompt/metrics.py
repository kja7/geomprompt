"""Dataset-level segmentation metrics and the paper's OHEM loss."""

import torch
from torch.nn import functional as F


def update_histogram(hist, prediction, label):
    valid = (label >= 0) & (label < hist.shape[0])
    bins = hist.shape[0] * label[valid] + prediction[valid]
    hist += torch.bincount(bins, minlength=hist.numel()).reshape_as(hist)


def summarize(hist):
    hist = hist.double()
    intersection = hist.diag()
    union = hist.sum(0) + hist.sum(1) - intersection
    present = union > 0
    iou = intersection / union.clamp_min(1)
    return {
        "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
        "pixel_accuracy": float(intersection.sum() / hist.sum().clamp_min(1)),
        "iou_per_class": iou.tolist(),
    }


def ohem_loss(logits, labels, threshold=0.6, min_kept=256):
    valid = labels != 255
    if not valid.any():
        return logits.sum() * 0
    with torch.no_grad():
        probabilities = (
            logits.softmax(1)
            .gather(1, labels.masked_fill(~valid, 0).unsqueeze(1))
            .squeeze(1)
        )
        values = probabilities[valid]
        if values.numel() >= min_kept:
            threshold = max(threshold, float(values.kthvalue(min_kept).values))
            valid = valid & (probabilities <= threshold)
    return F.cross_entropy(logits, labels.masked_fill(~valid, 255), ignore_index=255)
