import torch
import torch.nn as nn

from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D


class PairWaveletBatchTransform(nn.Module):
    def __init__(self, wave="haar", mode=None, J=1):
        super().__init__()
        if J != 1:
            raise ValueError(f"Only J=1 is supported for now, got {J}")
        if wave != "haar":
            raise ValueError(f"Only haar wave is supported for consistency with WaveDiff, got {wave}")

        self.wave = wave
        self.mode = mode
        self.J = J

        self.dwt = DWT_2D(wave)
        self.iwt = IDWT_2D(wave)

    def forward(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("batch must be a tuple or list like (x0, y, ...)")

        x0 = batch[0]
        y = batch[1]

        x0_ll, x0_lh, x0_hl, x0_hh = self.dwt(x0)
        y_ll, y_lh, y_hl, y_hh = self.dwt(y)

        x0_hf_cat = self.pack_hf(x0_lh, x0_hl, x0_hh)
        y_hf_cat = self.pack_hf(y_lh, y_hl, y_hh)

        x0_wavelet_cat = self.pack_wavelet(x0_ll, x0_lh, x0_hl, x0_hh)
        y_wavelet_cat = self.pack_wavelet(y_ll, y_lh, y_hl, y_hh)

        return {
            "x0": x0,
            "y": y,

            "x0_ll": x0_ll,
            "x0_lh": x0_lh,
            "x0_hl": x0_hl,
            "x0_hh": x0_hh,
            "x0_hf_cat": x0_hf_cat,
            "x0_wavelet_cat": x0_wavelet_cat,

            "y_ll": y_ll,
            "y_lh": y_lh,
            "y_hl": y_hl,
            "y_hh": y_hh,
            "y_hf_cat": y_hf_cat,
            "y_wavelet_cat": y_wavelet_cat,
        }

    @staticmethod
    def pack_hf(lh, hl, hh):
        return torch.cat([lh, hl, hh], dim=1)

    @staticmethod
    def unpack_hf(hf_cat):
        if hf_cat.shape[1] % 3 != 0:
            raise ValueError(f"hf_cat channels must be divisible by 3, got {hf_cat.shape[1]}")
        c = hf_cat.shape[1] // 3
        lh = hf_cat[:, 0:c]
        hl = hf_cat[:, c:2 * c]
        hh = hf_cat[:, 2 * c:3 * c]
        return lh, hl, hh

    @staticmethod
    def pack_wavelet(ll, lh, hl, hh):
        return torch.cat([ll, lh, hl, hh], dim=1)

    @staticmethod
    def unpack_wavelet(wavelet_cat):
        if wavelet_cat.shape[1] % 4 != 0:
            raise ValueError(f"wavelet_cat channels must be divisible by 4, got {wavelet_cat.shape[1]}")
        c = wavelet_cat.shape[1] // 4
        ll = wavelet_cat[:, 0:c]
        lh = wavelet_cat[:, c:2 * c]
        hl = wavelet_cat[:, 2 * c:3 * c]
        hh = wavelet_cat[:, 3 * c:4 * c]
        return ll, lh, hl, hh

    def inverse_from_parts(self, ll, lh, hl, hh):
        return self.iwt(ll, lh, hl, hh)

    def inverse_from_hf_cat(self, ll, hf_cat):
        lh, hl, hh = self.unpack_hf(hf_cat)
        return self.inverse_from_parts(ll, lh, hl, hh)

    def inverse_from_wavelet_cat(self, wavelet_cat):
        ll, lh, hl, hh = self.unpack_wavelet(wavelet_cat)
        return self.inverse_from_parts(ll, lh, hl, hh)