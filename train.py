"""Train GeomPrompt or GeomPrompt-Recovery on a frozen SUN RGB-D segmenter."""

import argparse
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from geomprompt.checkpoint import read_checkpoint, select_state_dict
from geomprompt.corruptions import (
    CORRUPTION_NAMES,
    apply_depth_corruption,
    sample_corruption,
)
from geomprompt.data import SUNRGBD, depth_tensor
from geomprompt.metrics import ohem_loss, summarize, update_histogram
from geomprompt.model import (
    GeomPrompt,
    GeomPromptRecovery,
    compute_magnitude_loss,
    compute_tv_loss,
)
from geomprompt.segmenters import build_segmenter, segmenter_logits


def broken_depth(clean, seed):
    rng = np.random.default_rng(seed)
    batch = []
    for image in clean.numpy():
        if rng.random() < 0.2:
            broken = image
        else:
            spec = sample_corruption(rng, CORRUPTION_NAMES, 0.1, 0.9)
            broken = apply_depth_corruption(image, spec, rng)
        batch.append(depth_tensor(broken))
    return torch.stack(batch)


def forward_prompt(model, batch, args, device, seed):
    rgb = batch["rgb"].to(device, non_blocking=True)
    if args.method == "recovery":
        depth = broken_depth(batch["depth_u8"], seed).to(device)
        prompt, delta, raw = model(rgb, depth)
    else:
        prompt, delta, raw = model(rgb)
    return rgb, prompt, delta, raw


@torch.no_grad()
def validate(model, segmenter, loader, args, device, rank):
    model.eval()
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    hist = torch.zeros(37, 37, device=device, dtype=torch.int64)
    for index, batch in enumerate(loader):
        rgb, prompt, _, _ = forward_prompt(
            model, batch, args, device, args.seed + rank * 1000000007 + index
        )
        label = batch["label"].to(device)
        logits = segmenter_logits(segmenter, args.segmenter, rgb, prompt)
        logits = F.interpolate(
            logits, size=label.shape[-2:], mode="bilinear", align_corners=False
        )
        update_histogram(hist, logits.argmax(1), label)
    if dist.is_initialized():
        dist.all_reduce(hist)
    model.train()
    return summarize(hist)["mean_iou"]


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, best, args):
    state = model.state_dict()
    ema_state = dict(state)
    ema_state.update(ema)
    torch.save(
        {
            "epoch": epoch,
            "best_val_miou": best,
            "model_state_dict": state,
            "ema_state_dict": ema_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "version": "v2",
                "segmenter": args.segmenter,
                "condition_on_broken_depth": args.method == "recovery",
                "image_size": args.image_size,
                "lowpass_factor": args.lowpass_factor,
                "use_adapter": not args.disable_adapter,
                "adapter_input_space": "dformer",
            },
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segmenter", choices=["dformer", "geminifusion"], required=True
    )
    parser.add_argument("--segmenter-checkpoint", required=True)
    parser.add_argument("--method", choices=["prompt", "recovery"], default="prompt")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-process microbatch size"
    )
    parser.add_argument(
        "--accum-steps", type=int, help="Default gives an effective global batch of 32"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--lr-encoder", type=float, default=3e-5)
    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--scale-start", type=float, default=15)
    parser.add_argument("--scale-end", type=float, default=80)
    parser.add_argument("--lambda-tv", type=float, default=1e-5)
    parser.add_argument("--lambda-mag", type=float, default=5e-4)
    parser.add_argument("--lowpass-factor", type=int, default=2)
    parser.add_argument("--freeze-vit", action="store_true")
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--num-samples", type=int, help="Limit training samples for smoke runs"
    )
    parser.add_argument("--val-samples", type=int, help="Limit full validation samples")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--quick-val-samples", type=int, default=300)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    for field in (
        "epochs",
        "batch_size",
        "val_every",
        "image_size",
        "lowpass_factor",
        "quick_val_samples",
    ):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    for field in ("num_samples", "val_samples", "accum_steps"):
        if getattr(args, field) is not None and getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.image_size % 32 or args.lowpass_factor > args.image_size:
        parser.error(
            "--image-size must be divisible by 32; --lowpass-factor cannot exceed it"
        )
    if args.warmup_epochs < 0 or args.num_workers < 0:
        parser.error("--warmup-epochs and --num-workers must be nonnegative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device)
    if world > 1:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    args.accum_steps = args.accum_steps or max(
        1, math.ceil(32 / (world * args.batch_size))
    )
    args.output_dir = (
        args.output_dir or Path("runs") / f"{args.method}_{args.segmenter}"
    )
    if (args.output_dir / "last.pth").exists() and args.resume is None:
        parser.error(
            "Output already contains a run; use --resume or a different --output-dir"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data = SUNRGBD(
        args.data_root,
        split="train",
        train=True,
        recovery=args.method == "recovery",
        segmenter=args.segmenter,
        image_size=args.image_size,
        num_samples=args.num_samples,
    )
    val_data = SUNRGBD(
        args.data_root,
        split="test",
        recovery=args.method == "recovery",
        segmenter="dformer",
        image_size=args.image_size,
        num_samples=args.val_samples,
    )
    # The trainer validates 480x480 crops; evaluate.py runs the full paper protocol.
    sampler = DistributedSampler(train_data, shuffle=True) if world > 1 else None
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loaders = []
    for count in [min(args.quick_val_samples, len(val_data)), len(val_data)]:
        subset = Subset(val_data, list(range(rank, count, world)))
        val_loaders.append(
            DataLoader(
                subset,
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
        )
    model_type = GeomPromptRecovery if args.method == "recovery" else GeomPrompt
    core = model_type(
        pretrained=args.pretrained and args.resume is None,
        freeze_vit=args.freeze_vit,
        image_size=args.image_size,
        residual_scale=args.scale_start,
        lowpass_factor=args.lowpass_factor,
        use_adapter=not args.disable_adapter,
    ).to(device)
    if args.disable_adapter:
        core.adapter.requires_grad_(False)
    groups = []
    for is_encoder, lr in [(True, args.lr_encoder), (False, args.lr_decoder)]:
        parameters = [
            p
            for name, p in core.named_parameters()
            if p.requires_grad and name.startswith("encoder.") == is_encoder
        ]
        if parameters:
            groups.append({"params": parameters, "lr": lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.accum_steps)
    total_steps, warmup_steps = (
        args.epochs * steps_per_epoch,
        args.warmup_epochs * steps_per_epoch,
    )

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(
            1, max(0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        )
        return (1 - progress) ** args.poly_power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ema = {
        name: p.detach().clone()
        for name, p in core.named_parameters()
        if p.requires_grad
    }
    start_epoch, best = 0, 0.0
    if args.resume:
        checkpoint = read_checkpoint(args.resume)
        if (
            checkpoint.get("config", {}).get("adapter_input_space", "dformer")
            != "dformer"
        ):
            parser.error("Resume requires a normalized-space training checkpoint")
        core.load_state_dict(select_state_dict(checkpoint, ema=False), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        ema = {name: checkpoint["ema_state_dict"][name].to(device) for name in ema}
        start_epoch, best = checkpoint["epoch"], checkpoint.get("best_val_miou", 0.0)
    model = (
        DDP(core, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1
        else core
    )
    segmenter = build_segmenter(args.segmenter, args.segmenter_checkpoint, device)
    if rank == 0:
        print(
            f"{args.method}/{args.segmenter}: effective batch={world * args.batch_size * args.accum_steps}, output={args.output_dir}"
        )
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        core.train()
        progress = epoch / max(1, args.epochs - 1)
        core.set_residual_scale(
            args.scale_start + (args.scale_end - args.scale_start) * progress
        )
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(2, device=device)
        iterator = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=rank != 0,
        )
        for index, batch in iterator:
            boundary = (index + 1) % args.accum_steps == 0 or index + 1 == len(loader)
            sync = model.no_sync() if world > 1 and not boundary else nullcontext()
            # Normalize the final partial accumulation group by its actual length.
            group_size = min(
                args.accum_steps,
                len(loader) - (index // args.accum_steps) * args.accum_steps,
            )
            with sync:
                seed = args.seed + rank * 1000000007 + epoch * 1000003 + index
                rgb, prompt, delta, raw = forward_prompt(
                    model, batch, args, device, seed
                )
                label = batch["label"].to(device)
                logits = segmenter_logits(segmenter, args.segmenter, rgb, prompt)
                logits = F.interpolate(
                    logits, size=label.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = (
                    ohem_loss(logits, label)
                    + args.lambda_tv * compute_tv_loss(raw)
                    + args.lambda_mag * compute_magnitude_loss(delta)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Nonfinite loss at epoch {epoch + 1}, batch {index}"
                    )
                (loss / group_size).backward()
            if boundary:
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                with torch.no_grad():
                    for name, parameter in core.named_parameters():
                        if name in ema:
                            ema[name].lerp_(parameter, 0.001)
            totals += torch.tensor([loss.detach().item(), 1], device=device)
            if rank == 0:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
        if world > 1:
            dist.all_reduce(totals)
        full = (
            epoch < 5 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        )
        backup = {
            name: p.detach().clone()
            for name, p in core.named_parameters()
            if name in ema
        }
        with torch.no_grad():
            for name, parameter in core.named_parameters():
                if name in ema:
                    parameter.copy_(ema[name])
        score = validate(core, segmenter, val_loaders[int(full)], args, device, rank)
        with torch.no_grad():
            for name, parameter in core.named_parameters():
                if name in backup:
                    parameter.copy_(backup[name])
        improved = full and (
            score > best or not (args.output_dir / "best.pth").exists()
        )
        best = max(best, score) if full else best
        if rank == 0:
            print(
                f"Loss={totals[0] / totals[1]:.4f} | {'full' if full else 'quick'} val mIoU={100 * score:.2f}%"
            )
            save_checkpoint(
                args.output_dir / "last.pth",
                core,
                ema,
                optimizer,
                scheduler,
                epoch + 1,
                best,
                args,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pth",
                    core,
                    ema,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    best,
                    args,
                )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
