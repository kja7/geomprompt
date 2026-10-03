"""SUN RGB-D loading and the preprocessing used by the experiment scripts."""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .corruptions import u16_depth_to_u8

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
TRAIN_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)


def rgb_tensor(rgb):
    # Use the original float64 numpy normalization before converting to float32.
    value = (rgb.astype(np.float64) / 255 - np.array(RGB_MEAN)) / np.array(RGB_STD)
    return torch.from_numpy(value.transpose(2, 0, 1).copy()).float()


def depth_tensor(depth):
    value = torch.from_numpy(depth.copy()).float().unsqueeze(0).expand(3, -1, -1)
    return (value / 255 - 0.48) / 0.28


def augment(rgb, label, depth, size):
    if random.random() < 0.5:
        rgb, label = rgb[:, ::-1].copy(), label[:, ::-1].copy()
        if depth is not None:
            depth = depth[:, ::-1].copy()
    scale = random.choice(TRAIN_SCALES)
    h, w = max(1, round(rgb.shape[0] * scale)), max(1, round(rgb.shape[1] * scale))
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    if depth is not None:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    top = random.randint(0, h - size) if h > size else 0
    left = random.randint(0, w - size) if w > size else 0
    rgb, label = (
        rgb[top : top + size, left : left + size],
        label[top : top + size, left : left + size],
    )
    if depth is not None:
        depth = depth[top : top + size, left : left + size]
    h, w = rgb.shape[:2]
    dh, dw = max(0, size - h), max(0, size - w)
    borders = (dh // 2, dh - dh // 2, dw // 2, dw - dw // 2)
    rgb = cv2.copyMakeBorder(rgb, *borders, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    label = cv2.copyMakeBorder(label, *borders, cv2.BORDER_CONSTANT, value=0)
    if depth is not None:
        depth = cv2.copyMakeBorder(depth, *borders, cv2.BORDER_CONSTANT, value=0)
    return rgb, label, depth


def geminifusion_preprocess(rgb, label, depth):
    """Paper evaluation: a single square 480x480 input, without TTA."""
    rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
    label = torch.from_numpy(label.copy()).long()
    rgb = TF.resize(rgb, (480, 480), InterpolationMode.BILINEAR)
    label = TF.resize(
        label.unsqueeze(0), (480, 480), InterpolationMode.NEAREST
    ).squeeze(0)
    rgb = TF.normalize(rgb / 255, RGB_MEAN, RGB_STD)
    if depth is not None:
        depth = torch.from_numpy(depth.transpose(2, 0, 1).copy()).float()
        depth = TF.resize(depth, (480, 480), InterpolationMode.BILINEAR)
        depth = (depth / 255 - 0.48) / 0.28
    return rgb, label, depth


class SUNRGBD(Dataset):
    def __init__(
        self,
        root=None,
        split="test",
        train=False,
        recovery=False,
        segmenter="dformer",
        image_size=480,
        num_samples=None,
        manifest=None,
        need_depth=False,
    ):
        self.root = Path(root).expanduser() if root else None
        self.train, self.recovery = train, recovery
        self.need_depth = need_depth or recovery
        self.segmenter, self.image_size = segmenter, image_size
        self.rows, self.hf = [], None
        if self.root is not None:
            path = (
                Path(manifest).expanduser() if manifest else self.root / f"{split}.txt"
            )
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = line.split()
                if len(row) != 3:
                    raise ValueError(f"{path}:{number}: expected RGB LABEL DEPTH paths")
                self.rows.append(row)
        else:
            if self.need_depth or segmenter == "geminifusion":
                raise ValueError(
                    "Use --data-root with local SUN RGB-D for recovery, depth baselines, or GeminiFusion"
                )
            from datasets import load_dataset

            self.hf = load_dataset("wyrx/SUNRGBD_seg", "uint8", split=split)
        length = len(self.rows) if self.hf is None else len(self.hf)
        self.length = min(length, num_samples) if num_samples is not None else length
        if self.length < 1:
            raise ValueError("SUN RGB-D split is empty")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        depth = None
        if self.hf is not None:
            sample = self.hf[index]
            rgb, label = np.array(sample["image"]), np.array(sample["label"])
        else:
            rgb_path, label_path, depth_path = [self.root / p for p in self.rows[index]]
            with Image.open(rgb_path) as image:
                rgb = np.array(image.convert("RGB"))
            with Image.open(label_path) as image:
                label = np.array(image)
            if self.need_depth:
                with Image.open(depth_path) as image:
                    raw_depth = np.array(image)
                depth = raw_depth[..., 0] if raw_depth.ndim == 3 else raw_depth
        if label.ndim == 3:
            label = label[..., 0]
        if not np.isin(label, np.r_[0:38, 255]).all():
            raise ValueError(
                "Expected SUN RGB-D labels 0 (void), 1..37; optionally 255 (void)"
            )
        if self.train:
            rgb, label, depth = augment(rgb, label, depth, self.image_size)
        if depth is not None and depth.dtype != np.uint8:
            depth = u16_depth_to_u8(depth.astype(np.uint16))
        label = np.where((label == 0) | (label == 255), 255, label.astype(np.int64) - 1)
        if not self.train and self.segmenter == "geminifusion":
            depth_3 = (
                np.repeat(depth[..., None], 3, axis=2) if depth is not None else None
            )
            rgb, label, depth_norm = geminifusion_preprocess(rgb, label, depth_3)
        else:
            if not self.train and self.image_size:
                target = (self.image_size, self.image_size)
                rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, target, interpolation=cv2.INTER_NEAREST)
                if depth is not None:
                    depth = cv2.resize(depth, target, interpolation=cv2.INTER_NEAREST)
            rgb = rgb_tensor(rgb)
            label = torch.from_numpy(label.copy()).long()
            depth_norm = depth_tensor(depth) if depth is not None else None
        result = {"rgb": rgb, "label": label, "index": index}
        if depth_norm is not None:
            result["depth"] = depth_norm
        if depth is not None:
            result["depth_u8"] = torch.from_numpy(depth.copy())
        return result
