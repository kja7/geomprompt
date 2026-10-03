"""Small PyTorch equivalents of the OpenMMLab layers used by the segmenters."""

# Adapted from OpenMMLab MMCV (Apache-2.0); see LICENSE_OPENMMLAB.
# Modified: retain only the convolution, normalization, and activation paths used here.
from torch import nn
from timm.layers import DropPath


def build_norm_layer(config, channels):
    layer = nn.SyncBatchNorm if config.get("type") == "SyncBN" else nn.BatchNorm2d
    return "bn", layer(channels)


def build_dropout(config):
    return DropPath(config["drop_prob"])


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        conv_cfg=None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, bias=norm_cfg is None
        )
        self.bn = (
            build_norm_layer(norm_cfg, out_channels)[1] if norm_cfg else nn.Identity()
        )
        self.activate = nn.ReLU(inplace=True) if act_cfg else nn.Identity()

    def forward(self, value):
        return self.activate(self.bn(self.conv(value)))
