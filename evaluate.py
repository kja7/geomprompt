"""Evaluate GeomPrompt, GeomPrompt-Recovery, and depth controls on SUN RGB-D."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from geomprompt.checkpoint import load_prompt
from geomprompt.corruptions import (
    CORRUPTION_NAMES,
    CorruptionSpec,
    apply_depth_corruption,
)
from geomprompt.data import SUNRGBD, depth_tensor
from geomprompt.metrics import summarize, update_histogram
from geomprompt.segmenters import build_segmenter, segmenter_logits


def predict(segmenter, name, rgb, depth, prompter=None, recovery=False, tta=True):
    height, width = rgb.shape[-2:]
    scores = rgb.new_zeros((rgb.shape[0], 37, height, width))
    scales = (0.5, 0.75, 1.0, 1.25, 1.5) if name == "dformer" and tta else (1.0,)
    flips = (False, True) if name == "dformer" and tta else (False,)
    for scale in scales:
        size = (max(1, int(height * scale)), max(1, int(width * scale)))
        rgb_scaled = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
        depth_scaled = (
            F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
            if depth is not None
            else None
        )
        for flip in flips:
            image = rgb_scaled.flip(-1) if flip else rgb_scaled
            geometry = (
                depth_scaled.flip(-1)
                if flip and depth_scaled is not None
                else depth_scaled
            )
            pad = (0, (-size[1]) % 32, 0, (-size[0]) % 32)
            image = F.pad(image, pad)
            if geometry is not None:
                geometry = F.pad(geometry, pad)
            if prompter is not None:
                geometry = (
                    prompter(image, geometry)[0] if recovery else prompter(image)[0]
                )
            logits = segmenter_logits(segmenter, name, image, geometry)
            logits = F.interpolate(
                logits, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            logits = logits[..., : size[0], : size[1]]
            if flip:
                logits = logits.flip(-1)
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            scores += logits.softmax(1)
    return scores.argmax(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segmenter", choices=["dformer", "geminifusion"], required=True
    )
    parser.add_argument("--segmenter-checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=["prompt", "recovery", "rgb", "gt", "broken"],
        default="prompt",
    )
    parser.add_argument("--prompt-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest")
    parser.add_argument("--corruption", choices=CORRUPTION_NAMES, default="noise")
    parser.add_argument("--severity", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable DFormer multi-scale/flip inference",
    )
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--output", type=Path, help="Save metrics as JSON")
    args = parser.parse_args()
    if args.mode in ("prompt", "recovery") and args.prompt_checkpoint is None:
        parser.error("--prompt-checkpoint is required for prompt/recovery mode")
    if not 0 <= args.severity <= 1:
        parser.error("--severity must be between 0 and 1")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.manifest and not args.data_root:
        parser.error("--manifest requires --data-root")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    segmenter = build_segmenter(args.segmenter, args.segmenter_checkpoint, device)
    prompter = None
    if args.mode in ("prompt", "recovery"):
        prompter = load_prompt(
            args.prompt_checkpoint,
            args.segmenter,
            recovery=args.mode == "recovery",
            ema=not args.no_ema,
        ).to(device)
    dataset = SUNRGBD(
        root=args.data_root,
        manifest=args.manifest,
        segmenter=args.segmenter,
        recovery=args.mode == "recovery",
        need_depth=args.mode in ("gt", "broken", "recovery"),
        image_size=None,
        num_samples=args.num_samples,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    hist = torch.zeros(37, 37, dtype=torch.int64, device=device)
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.segmenter}/{args.mode}"):
            rgb, label = batch["rgb"].to(device), batch["label"].to(device)
            if args.mode == "rgb":
                depth = torch.full_like(rgb, -0.48 / 0.28)
            elif args.mode == "gt":
                depth = batch["depth"].to(device)
            elif args.mode in ("broken", "recovery"):
                index = int(batch["index"][0])
                rng = np.random.default_rng(args.seed + index * 10007)
                broken = apply_depth_corruption(
                    batch["depth_u8"][0].numpy(),
                    CorruptionSpec(args.corruption, args.severity),
                    rng,
                )
                depth = depth_tensor(broken).unsqueeze(0).to(device)
                if args.segmenter == "geminifusion":
                    depth = F.interpolate(
                        depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False
                    )
            else:
                depth = None
            prediction = predict(
                segmenter,
                args.segmenter,
                rgb,
                depth,
                prompter,
                recovery=args.mode == "recovery",
                tta=not args.no_tta,
            )
            update_histogram(hist, prediction, label)
    result = dict(
        segmenter=args.segmenter,
        mode=args.mode,
        samples=len(dataset),
        **summarize(hist),
    )
    if args.mode in ("broken", "recovery"):
        result.update(corruption=args.corruption, severity=args.severity)
    print(
        f"mIoU: {100 * result['mean_iou']:.2f}% | PA: {100 * result['pixel_accuracy']:.2f}%"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
