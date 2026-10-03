"""Load the original experiment checkpoints and release checkpoints."""

from pathlib import Path

import torch

from .model import GeomPrompt, GeomPromptRecovery


def read_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def select_state_dict(checkpoint, ema=True):
    keys = ["ema_state_dict", "model_ema_state_dict"] if ema else []
    keys += ["model_state_dict", "model", "state_dict", "segmenter"]
    for key in keys:
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        raise ValueError("Checkpoint does not contain a model state dict")
    return {k.removeprefix("module."): v for k, v in checkpoint.items()}


def load_prompt(path, segmenter, recovery=False, ema=True):
    checkpoint = read_checkpoint(path)
    config = checkpoint.get("config", {})
    settings = checkpoint.get("args", {})
    state = select_state_dict(checkpoint, ema=ema)
    is_recovery = any(k.startswith("depth_encoder.") for k in state)
    if is_recovery != recovery:
        raise ValueError(
            "Prompt checkpoint does not match the requested prompt/recovery mode"
        )
    if "upsampler.blur1.weight" not in state:
        raise ValueError("Only paper GeomPrompt checkpoints (V2) are supported")
    size = config.get("image_size", 480)
    if isinstance(size, (tuple, list)):
        size = tuple(size)
    kwargs = dict(
        pretrained=False,
        image_size=size,
        dynamic_img_size=True,
        residual_scale=float(config.get("scale_end", 80)),
        lowpass_factor=int(config.get("lowpass_factor", 2)),
        use_adapter=bool(
            config.get("use_adapter", not settings.get("disable_adapter", False))
        ),
    )
    if recovery:
        model = GeomPromptRecovery(**kwargs)
    else:
        default_space = "dformer" if segmenter == "dformer" else "unit"
        # Express older unit-space adapters in normalized depth coordinates.
        # This preserves their learned function while using the paper architecture.
        if config.get("adapter_input_space", default_space) == "unit":
            first_weight = state["adapter.conv1.weight"]
            state["adapter.conv1.bias"] = state[
                "adapter.conv1.bias"
            ] + 0.48 * first_weight.sum((1, 2, 3))
            state["adapter.conv1.weight"] = first_weight * 0.28
            state["adapter.conv3.weight"] = state["adapter.conv3.weight"] / 0.28
            state["adapter.conv3.bias"] = state["adapter.conv3.bias"] / 0.28
        model = GeomPrompt(**kwargs)
    model.load_state_dict(state, strict=True)
    return model.eval()
