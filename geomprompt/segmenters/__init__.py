"""Frozen SUN RGB-D segmenters used in the paper."""

from torch import nn
from torch.nn import functional as F


class DFormer(nn.Module):
    def __init__(self):
        super().__init__()
        from .dformer_backbone import DFormer_Base
        from .dformer_head import LightHamHead

        norm = dict(type="BN", requires_grad=True)
        self.backbone = DFormer_Base(drop_path_rate=0.1, norm_cfg=norm)
        self.decode_head = LightHamHead(
            in_channels=[128, 256, 512],
            in_index=[1, 2, 3],
            channels=512,
            num_classes=37,
            norm_cfg=norm,
        )

    def forward(self, rgb, depth):
        features, _ = self.backbone(rgb, depth)
        logits = self.decode_head(features)
        return F.interpolate(
            logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False
        )


def build_segmenter(name, checkpoint, device):
    from ..checkpoint import read_checkpoint, select_state_dict

    if name == "dformer":
        model = DFormer()
    elif name == "geminifusion":
        from .geminifusion import GeminiFusion

        model = GeminiFusion()
    else:
        raise ValueError(f"Unknown segmenter: {name}")
    state = select_state_dict(read_checkpoint(checkpoint), ema=False)
    if name == "dformer":
        # Released DFormer weights include output norms unused by its forward pass.
        unused = {
            f"backbone.norm{i}.{field}"
            for i in range(4)
            for field in ("weight", "bias")
        }
        state = {key: value for key, value in state.items() if key not in unused}
    else:
        # Official MiT-B3 checkpoints contain a pruning head unused in inference.
        state = {
            key: value
            for key, value in state.items()
            if not key.startswith("encoder.score_predictor.")
        }
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).to(device).eval()


def segmenter_logits(model, name, rgb, prompt):
    if name == "dformer":
        return model(rgb.flip(1), prompt)
    depth = (prompt * 0.28 + 0.48).clamp(0, 1)
    return model(rgb, depth)
