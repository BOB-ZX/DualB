from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .ncsnpp_generator_adagn import WaveletNCSNpp


class WaveDiffNCSNppAdapter(nn.Module):
    """Thin SelfRDB-compatible wrapper around WaveDiff official WaveletNCSNpp.

    The official generator has forward(x, time_cond, z). SelfRDB-style runners
    often call generator(x, t, z=None, x_r=None). This adapter preserves the
    official backbone and only adapts the interface.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.config = self._build_config(kwargs)
        self.nz = int(self.config.nz)
        self.concat_x_r = bool(getattr(self.config, "concat_x_r", False))
        self.model = WaveletNCSNpp(self.config)

    @staticmethod
    def _build_config(kwargs: dict[str, Any]) -> SimpleNamespace:
        defaults = {
            "image_size": 128,
            "num_channels": 8,
            "num_out_channels": 4,
            "num_channels_dae": 64,
            "ch_mult": [1, 2, 2, 2],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "resamp_with_conv": True,
            "conditional": True,
            "fir": True,
            "fir_kernel": [1, 3, 3, 1],
            "skip_rescale": True,
            "resblock_type": "biggan",
            "progressive": "none",
            "progressive_input": "residual",
            "progressive_combine": "sum",
            "embedding_type": "positional",
            "fourier_scale": 16.0,
            "patch_size": 1,
            "not_use_tanh": False,
            "z_emb_dim": 256,
            "nz": 100,
            "n_mlp": 3,
            "centered": True,
            "no_use_fbn": False,
            "no_use_freq": False,
            "no_use_residual": False,
            "concat_x_r": False,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def forward(self, x: Tensor, t: Tensor, z: Tensor | None = None, x_r: Tensor | None = None) -> Tensor:
        if self.concat_x_r:
            if x_r is None:
                raise ValueError("concat_x_r=True requires x_r")
            x = torch.cat([x_r, x], dim=1)

        if z is None:
            z = torch.randn(x.shape[0], self.nz, device=x.device, dtype=x.dtype)
        if t.ndim != 1:
            t = t.reshape(x.shape[0])

        if self.config.embedding_type == "fourier":
            time_cond = t.to(device=x.device, dtype=x.dtype)
            time_cond = torch.clamp(time_cond, min=1).div(float(max(getattr(self.config, "num_timesteps", 1000), 1)))
        else:
            time_cond = t.to(device=x.device, dtype=torch.long)

        return self.model(x, time_cond, z)


WaveDiffNCSNpp = WaveDiffNCSNppAdapter
