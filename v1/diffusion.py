from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _extract(values: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
    out = torch.gather(values, 0, t)
    return out.reshape([shape[0]] + [1] * (len(shape) - 1))


class HFDiffusionCoefficients:
    """WaveDiff/DDGAN forward coefficients for high-frequency subbands."""

    def __init__(
        self,
        n_steps: int,
        device: torch.device,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        use_geometric: bool = False,
    ) -> None:
        self.n_steps = int(n_steps)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.use_geometric = bool(use_geometric)
        self.sigmas, self.a_s, _ = self.get_sigma_schedule(device)
        self.a_s_cum = torch.cumprod(self.a_s, dim=0)
        self.sigmas_cum = torch.sqrt(torch.clamp(1.0 - self.a_s_cum ** 2, min=0.0))
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1.0

    @staticmethod
    def var_func_vp(t: Tensor, beta_min: float, beta_max: float) -> Tensor:
        log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
        return 1.0 - torch.exp(2.0 * log_mean_coeff)

    @staticmethod
    def var_func_geometric(t: Tensor, beta_min: float, beta_max: float) -> Tensor:
        return beta_min * ((beta_max / beta_min) ** t)

    def get_sigma_schedule(self, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
        eps_small = 1e-3
        t = np.arange(0, self.n_steps + 1, dtype=np.float64) / self.n_steps
        t = torch.from_numpy(t).to(device=device, dtype=torch.float32)
        t = t * (1.0 - eps_small) + eps_small
        if self.use_geometric:
            var = self.var_func_geometric(t, self.beta_min, self.beta_max)
        else:
            var = self.var_func_vp(t, self.beta_min, self.beta_max)
        alpha_bars = 1.0 - var
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.cat([torch.tensor([1e-8], device=device), betas]).float()
        sigmas = torch.sqrt(torch.clamp(betas, min=0.0))
        a_s = torch.sqrt(torch.clamp(1.0 - betas, min=0.0))
        return sigmas, a_s, betas

    def q_sample(self, x_start: Tensor, u: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return _extract(self.a_s_cum, u, x_start.shape) * x_start + _extract(self.sigmas_cum, u, x_start.shape) * noise

    def q_sample_pairs(self, x_start: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        noise = torch.randn_like(x_start)
        x_u = self.q_sample(x_start, u)
        x_up1 = _extract(self.a_s, u + 1, x_start.shape) * x_u + _extract(self.sigmas, u + 1, x_start.shape) * noise
        return x_u, x_up1


class HFPosteriorCoefficients:
    """WaveDiff/DDGAN posterior coefficients for high-frequency subbands."""

    def __init__(
        self,
        n_steps: int,
        device: torch.device,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        use_geometric: bool = False,
    ) -> None:
        tmp = HFDiffusionCoefficients(n_steps, device, beta_min=beta_min, beta_max=beta_max, use_geometric=use_geometric)
        betas = tmp.get_sigma_schedule(device)[2].float()[1:]
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

    def sample_posterior(self, x0_hat: Tensor, x_up1: Tensor, u: Tensor) -> Tensor:
        mean = _extract(self.posterior_mean_coef1, u, x_up1.shape) * x0_hat + _extract(
            self.posterior_mean_coef2, u, x_up1.shape
        ) * x_up1
        log_var = _extract(self.posterior_log_variance_clipped, u, x_up1.shape)
        noise = torch.randn_like(x_up1)
        nonzero = (1.0 - (u == 0).float()).view([x_up1.shape[0]] + [1] * (x_up1.ndim - 1))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise


class LLDiffusionBridge(nn.Module):
    """SelfRDB-style diffusion bridge applied to LL subbands."""

    def __init__(
        self,
        n_steps: int = 10,
        gamma: float = 1.0,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        n_recursions: int = 1,
        consistency_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.n_recursions = int(n_recursions)
        self.consistency_threshold = float(consistency_threshold)

        betas = self._get_betas(self.n_steps, self.beta_start, self.beta_end)
        s = np.cumsum(betas) ** 0.5
        s_bar = np.flip(np.cumsum(betas)) ** 0.5
        mu_x0, mu_y, _ = self.gaussian_product(s, s_bar)
        gamma_scaled = self.gamma * betas.sum()
        std = gamma_scaled * s / (s ** 2 + s_bar ** 2)

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))
        self.register_buffer("mu_x0", torch.tensor(mu_x0, dtype=torch.float32))
        self.register_buffer("mu_y", torch.tensor(mu_y, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    @staticmethod
    def _get_betas(n_steps: int, beta_start: float, beta_end: float) -> np.ndarray:
        beta_end = beta_end / n_steps
        betas_len = n_steps + 1
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, betas_len) ** 2
        betas = np.append(0.0, betas).astype(np.float32)
        if betas_len % 2 == 1:
            return np.concatenate([betas[: betas_len // 2], [betas[betas_len // 2]], np.flip(betas[: betas_len // 2])])
        return np.concatenate([betas[: betas_len // 2], np.flip(betas[: betas_len // 2])])

    @staticmethod
    def gaussian_product(sigma1: np.ndarray, sigma2: np.ndarray):
        denom = sigma1 ** 2 + sigma2 ** 2
        mu1 = sigma2 ** 2 / denom
        mu2 = sigma1 ** 2 / denom
        var = sigma1 ** 2 * sigma2 ** 2 / denom
        return mu1, mu2, var

    @staticmethod
    def _shape(coeff: Tensor, x: Tensor) -> Tensor:
        return coeff.view([-1] + [1] * (x.ndim - 1))

    def q_sample(self, t: Tensor, x0_ll: Tensor, y_ll: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if x0_ll.shape != y_ll.shape:
            raise ValueError(f"x0_ll and y_ll shape mismatch: {tuple(x0_ll.shape)} vs {tuple(y_ll.shape)}")
        if noise is None:
            noise = torch.randn_like(x0_ll)
        mu_x0 = self._shape(self.mu_x0[t], x0_ll)
        mu_y = self._shape(self.mu_y[t], x0_ll)
        std = self._shape(self.std[t], x0_ll)
        return mu_x0 * x0_ll + mu_y * y_ll + std * noise

    def q_posterior(
        self,
        t: Tensor,
        x_t_ll: Tensor,
        x0_hat_ll: Tensor,
        y_ll: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_t_ll)

        std_t = self._shape(self.s[t], x_t_ll)
        std_tm1 = self._shape(self.s[t - 1], x_t_ll)
        mu_x0_t = self._shape(self.mu_x0[t], x_t_ll)
        mu_x0_tm1 = self._shape(self.mu_x0[t - 1], x_t_ll)
        mu_y_t = self._shape(self.mu_y[t], x_t_ll)
        mu_y_tm1 = self._shape(self.mu_y[t - 1], x_t_ll)

        var_t = std_t ** 2
        var_tm1 = std_tm1 ** 2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / (mu_x0_tm1 + 1e-12)) ** 2
        v = var_t_tm1 * (var_tm1 / (var_t + 1e-12))
        mean = (
            mu_x0_tm1 * x0_hat_ll
            + mu_y_tm1 * y_ll
            + torch.sqrt(torch.clamp((var_tm1 - v) / (var_t + 1e-12), min=0.0))
            * (x_t_ll - mu_x0_t * x0_hat_ll - mu_y_t * y_ll)
        )
        return mean + torch.sqrt(torch.clamp(v, min=0.0)) * noise


class HybridWaveletDiffusion(nn.Module):
    """LL bridge and WaveDiff-style HF diffusion for a full wavelet state."""

    def __init__(
        self,
        n_steps: int = 10,
        gamma: float = 1.0,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        n_recursions: int = 1,
        consistency_threshold: float = 0.0,
        base_channels: int = 1,
        hf_beta_min: float = 0.1,
        hf_beta_max: float = 20.0,
        hf_use_geometric: bool = False,
    ) -> None:
        super().__init__()
        self.n_steps = int(n_steps)
        self.n_recursions = int(n_recursions)
        self.consistency_threshold = float(consistency_threshold)
        self.base_channels = int(base_channels)
        self.hf_beta_min = float(hf_beta_min)
        self.hf_beta_max = float(hf_beta_max)
        self.hf_use_geometric = bool(hf_use_geometric)
        self.ll_bridge = LLDiffusionBridge(
            n_steps=n_steps,
            gamma=gamma,
            beta_start=beta_start,
            beta_end=beta_end,
            n_recursions=n_recursions,
            consistency_threshold=consistency_threshold,
        )
        self._hf_device: torch.device | None = None
        self._hf_coeff: HFDiffusionCoefficients | None = None
        self._hf_pos_coeff: HFPosteriorCoefficients | None = None

    def _ensure_hf(self, device: torch.device) -> None:
        if self._hf_device == device and self._hf_coeff is not None and self._hf_pos_coeff is not None:
            return
        self._hf_coeff = HFDiffusionCoefficients(
            self.n_steps,
            device,
            beta_min=self.hf_beta_min,
            beta_max=self.hf_beta_max,
            use_geometric=self.hf_use_geometric,
        )
        self._hf_pos_coeff = HFPosteriorCoefficients(
            self.n_steps,
            device,
            beta_min=self.hf_beta_min,
            beta_max=self.hf_beta_max,
            use_geometric=self.hf_use_geometric,
        )
        self._hf_device = device

    def split_wavelet(self, wavelet: Tensor) -> Tuple[Tensor, Tensor]:
        c = self.base_channels
        if wavelet.shape[1] != 4 * c:
            raise ValueError(f"Expected {4 * c} wavelet channels, got {wavelet.shape[1]}")
        return wavelet[:, :c], wavelet[:, c:]

    @staticmethod
    def merge_wavelet(ll: Tensor, hf: Tensor) -> Tensor:
        return torch.cat([ll, hf], dim=1)

    def q_sample_mixed_pair(self, t: Tensor, x0_wavelet: Tensor, y_wavelet: Tensor) -> Tuple[Tensor, Tensor]:
        self._ensure_hf(x0_wavelet.device)
        assert self._hf_coeff is not None
        x0_ll, x0_hf = self.split_wavelet(x0_wavelet)
        y_ll, _ = self.split_wavelet(y_wavelet)
        u = t - 1

        x_t_ll = self.ll_bridge.q_sample(t, x0_ll, y_ll)
        x_tm1_ll = self.ll_bridge.q_posterior(t, x_t_ll, x0_ll, y_ll)
        x_tm1_hf, x_t_hf = self._hf_coeff.q_sample_pairs(x0_hf, u)
        return self.merge_wavelet(x_tm1_ll, x_tm1_hf), self.merge_wavelet(x_t_ll, x_t_hf)

    def q_posterior(self, t: Tensor, x_t_wavelet: Tensor, x0_hat_wavelet: Tensor, y_wavelet: Tensor) -> Tensor:
        self._ensure_hf(x_t_wavelet.device)
        assert self._hf_pos_coeff is not None
        x_t_ll, x_t_hf = self.split_wavelet(x_t_wavelet)
        x0_hat_ll, x0_hat_hf = self.split_wavelet(x0_hat_wavelet)
        y_ll, _ = self.split_wavelet(y_wavelet)
        u = t - 1
        x_tm1_ll = self.ll_bridge.q_posterior(t, x_t_ll, x0_hat_ll, y_ll)
        x_tm1_hf = self._hf_pos_coeff.sample_posterior(x0_hat_hf, x_t_hf, u)
        return self.merge_wavelet(x_tm1_ll, x_tm1_hf)

    @torch.inference_mode()
    def sample_wavelet_x0(
        self,
        y_wavelet: Tensor,
        predict_x0_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> Tensor:
        self._ensure_hf(y_wavelet.device)
        y_ll, y_hf = self.split_wavelet(y_wavelet)
        b = y_wavelet.shape[0]
        t_init = torch.full((b,), self.n_steps, device=y_wavelet.device, dtype=torch.long)
        x_t_ll = self.ll_bridge.q_sample(t_init, torch.zeros_like(y_ll), y_ll)
        x_t_hf = torch.randn_like(y_hf)
        x_t = self.merge_wavelet(x_t_ll, x_t_hf)
        pred = torch.zeros_like(y_wavelet)
        for step in range(self.n_steps, 0, -1):
            t = torch.full((b,), step, device=y_wavelet.device, dtype=torch.long)
            pred = predict_x0_fn(x_t, y_wavelet, t)
            x_t = self.q_posterior(t, x_t, pred, y_wavelet).detach()
        return pred


DiffusionBridge = HybridWaveletDiffusion
