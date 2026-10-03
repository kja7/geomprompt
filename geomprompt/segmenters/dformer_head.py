# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from VCIP-RGBD/DFormer and OpenMMLab MMSegmentation.
# Modified: inference-only decode head and device-independent NMF bases.
# See LICENSE_DFORMER and LICENSE_OPENMMLAB.
import torch
from torch import nn
from torch.nn import functional as F
from .layers import ConvModule

resize = F.interpolate


class BaseDecodeHead(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        in_index,
        norm_cfg=None,
        input_transform=None,
        dropout_ratio=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index
        self.conv_cfg = None
        self.norm_cfg = norm_cfg
        self.act_cfg = dict(type="ReLU")
        self.align_corners = False
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def _transform_inputs(self, inputs):
        return [inputs[i] for i in self.in_index]

    def cls_seg(self, feature):
        return self.conv_seg(self.dropout(feature))


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()
        self.spatial = args.setdefault("SPATIAL", True)
        self.S = args.setdefault("MD_S", 1)
        self.D = args.setdefault("MD_D", 512)
        self.R = args.setdefault("MD_R", 64)
        self.train_steps = args.setdefault("TRAIN_STEPS", 6)
        self.eval_steps = args.setdefault("EVAL_STEPS", 7)
        self.inv_t = args.setdefault("INV_T", 100)
        self.eta = args.setdefault("ETA", 0.9)
        self.rand_init = args.setdefault("RAND_INIT", True)

    def _build_bases(self, B, S, D, R):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return (bases, coef)

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)
        if not self.rand_init and (not hasattr(self, "bases")):
            bases = self._build_bases(1, self.S, D, self.R).to(x.device)
            self.register_buffer("bases", bases)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R).to(x.device)
        else:
            bases = self.bases.to(x.device).repeat(B, 1, 1)
        bases, coef = self.local_inference(x, bases)
        coef = self.compute_coef(x, bases, coef)
        x = torch.bmm(bases, coef.transpose(1, 2))
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)
        bases = bases.view(B, self.S, D, self.R)
        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)
        self.inv_t = 1

    def _build_bases(self, B, S, D, R):
        bases = torch.rand((B * S, D, R))
        bases = F.normalize(bases, dim=1)
        return bases

    def local_step(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-06)
        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-06)
        return (bases, coef)

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-06)
        return coef


class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), norm_cfg=None, **kwargs):
        super().__init__()
        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None
        )
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None
        )

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham


class LightHamHead(BaseDecodeHead):
    def __init__(self, ham_channels=512, ham_kwargs=dict(), **kwargs):
        super(LightHamHead, self).__init__(input_transform="multiple_select", **kwargs)
        self.ham_channels = ham_channels
        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.hamburger = Hamburger(ham_channels, ham_kwargs, **kwargs)
        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        inputs = [
            resize(
                level,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for level in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)
        output = self.cls_seg(output)
        return output
