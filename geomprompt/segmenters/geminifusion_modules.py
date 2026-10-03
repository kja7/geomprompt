# Adapted from JiaDingCN/GeminiFusion (MIT); see LICENSE_GEMINIFUSION.
from torch import nn

num_parallel = 2


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, "ln_" + str(i), nn.LayerNorm(num_features, eps=1e-06))

    def forward(self, x_parallel):
        return [getattr(self, "ln_" + str(i))(x) for i, x in enumerate(x_parallel)]
