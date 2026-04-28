from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusion import HybridWaveletDiffusion
from backbones.wavediff_adapter import WaveDiffNCSNpp
from backbones.discriminator import Discriminator_large
from datasets import DataModule
from utils import compute_metrics, save_eval_images, save_image_pair, save_preds
from waveletTransform import PairWaveletBatchTransform


class BridgeRunner(L.LightningModule):
    """SelfRDB Lightning runner using a WaveDiff-style wavelet generator.

    Full wavelet state order is [LL, LH, HL, HH]. LL uses the bridge process.
    HF subbands use WaveDiff/DDGAN forward and posterior coefficients.
    """

    def __init__(
        self,
        generator_params,
        discriminator_params,
        diffusion_params,
        lr_g: float,
        lr_d: float,
        disc_grad_penalty_freq: int,
        disc_grad_penalty_weight: float,
        lambda_rec_loss: float,
        optim_betas,
        eval_mask: bool,
        eval_subject: bool,
        lambda_adv_loss: float = 1.0,
        lambda_ll_loss: float = 1.0,
        lambda_hf_loss: float = 0.5,
        wavelet_scale: float = 1.0,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.lr_g = lr_g
        self.lr_d = lr_d
        self.disc_grad_penalty_freq = int(disc_grad_penalty_freq)
        self.disc_grad_penalty_weight = float(disc_grad_penalty_weight)
        self.lambda_rec_loss = float(lambda_rec_loss)
        self.lambda_adv_loss = float(lambda_adv_loss)
        self.lambda_ll_loss = float(lambda_ll_loss)
        self.lambda_hf_loss = float(lambda_hf_loss)
        self.optim_betas = tuple(optim_betas)
        self.eval_mask = bool(eval_mask)
        self.eval_subject = bool(eval_subject)
        self.wavelet_scale = float(wavelet_scale)

        self.n_steps = int(diffusion_params["n_steps"])
        self.n_recursions = int(diffusion_params.get("n_recursions", 1))
        self.consistency_threshold = float(diffusion_params.get("consistency_threshold", 0.0))
        self.base_channels = int(diffusion_params.get("base_channels", 1))
        self.wavelet_channels = 4 * self.base_channels

        generator_params = dict(generator_params)
        use_recursive_state = bool(generator_params.get("concat_x_r", False))
        generator_params["num_channels"] = (3 if use_recursive_state else 2) * self.wavelet_channels
        generator_params["num_out_channels"] = (3 if use_recursive_state else 2) * self.wavelet_channels
        self.latent_dim = int(latent_dim or generator_params.get("nz", 100))
        self.generator = WaveDiffNCSNpp(**generator_params)


        discriminator_params = dict(discriminator_params)
        discriminator_params.setdefault("nc", 2 * self.wavelet_channels)
        self.discriminator = Discriminator_large(**discriminator_params)

        self.diffusion = HybridWaveletDiffusion(**dict(diffusion_params))
        self.wavelet_transform = PairWaveletBatchTransform(
            wave="haar",
            mode="zero",
            J=1,
            wavelet_scale=self.wavelet_scale,
        )

    @staticmethod
    def adversarial_loss(pred: Tensor, is_real: bool) -> Tensor:
        loss = F.softplus(-pred) if is_real else F.softplus(pred)
        return loss.mean()

    def _latent(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)

    def _split(self, wavelet: Tensor) -> Tuple[Tensor, Tensor]:
        c = self.base_channels
        return wavelet[:, :c], wavelet[:, c:]

    def _predict_wavelet_x0(self, x_t_wavelet: Tensor, y_wavelet: Tensor, t: Tensor) -> Tensor:
        gen_input = torch.cat([x_t_wavelet.detach(), y_wavelet], dim=1)
        x0_r = torch.zeros_like(x_t_wavelet)
        pred = x0_r
        # z = self._latent(x_t_wavelet.shape[0], x_t_wavelet.device, x_t_wavelet.dtype)

        for _ in range(max(self.n_recursions, 1)):
            pred_next = self.generator(gen_input, t, x_r=x0_r)
            if self.consistency_threshold > 0.0:
                change = torch.abs(pred_next - x0_r).mean(dim=(1, 2, 3)).max()
                x0_r = pred_next
                pred = pred_next
                if change < self.consistency_threshold:
                    break
            else:
                x0_r = pred_next
                pred = pred_next
        return pred

    def _inverse_wavelet(self, wavelet: Tensor, out_size) -> Tensor:
        return self.wavelet_transform.inverse_from_cat(
            wavelet,
            base_channels=self.base_channels,
            output_size=out_size,
        ).clamp(-1.0, 1.0)

    def training_step(self, batch, batch_idx=None):
        x0, y, _ = batch # [bs, 1, 256, 256]
        optimizer_g, optimizer_d = self.optimizers()
        out = self.wavelet_transform(batch)
        x0_wavelet = out["x0_wavelet"]
        y_wavelet = out["y_wavelet"]
        out_size = x0.shape[-2:]

        # =========================
        # 1. Train D, LL only
        # =========================
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad(set_to_none=True)

        t = torch.randint(
            1, self.n_steps + 1,
            (x0_wavelet.shape[0],),
            device=x0_wavelet.device
        )

        x_prev_real, x_t = self.diffusion.q_sample_mixed_pair(t, x0_wavelet, y_wavelet)
        x_prev_real_for_d = x_prev_real.detach().requires_grad_(True)
        x_t_cond = x_t.detach()
        disc_real = self.discriminator(x_prev_real_for_d, t, x_t_cond)
        
        d_real_loss = self.adversarial_loss(disc_real, is_real=True)
        d_real_acc = (disc_real > 0).float().mean()

        gp_loss = torch.tensor(0.0, device=x0.device)
        if self.disc_grad_penalty_freq > 0 and self.global_step % self.disc_grad_penalty_freq == 0:
            grads = torch.autograd.grad(
                outputs=disc_real.sum(),
                inputs=x_prev_real_for_d,
                create_graph=True
            )[0]
            gp_loss = grads.view(grads.size(0), -1).norm(2, dim=1).pow(2).mean()
            gp_loss = 0.5 * self.disc_grad_penalty_weight * gp_loss
            d_real_loss = d_real_loss + gp_loss

        with torch.no_grad():
            x0_pred_detached = self._predict_wavelet_x0(x_t, y_wavelet, t).detach()
            x_prev_fake = self.diffusion.q_posterior(t, x_t, x0_pred_detached, y_wavelet)

        disc_fake = self.discriminator(x_prev_fake.detach(), t, x_t_cond)
        d_fake_loss = self.adversarial_loss(disc_fake, is_real=False)
        d_fake_acc = (disc_fake < 0).float().mean()

        d_loss = d_real_loss + d_fake_loss
        d_acc = 0.5 * (d_real_acc + d_fake_acc)
        d_loss = d_loss*1.0

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # =========================
        # 2. Train G
        # =========================
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad(set_to_none=True)
        
        t = torch.randint(
            1, self.n_steps + 1,
            (x0_wavelet.shape[0],),
            device=x0_wavelet.device
        )

        x_prev_real, x_t = self.diffusion.q_sample_mixed_pair(t, x0_wavelet, y_wavelet)
        x0_pred = self._predict_wavelet_x0(x_t, y_wavelet, t)
        x_prev_fake = self.diffusion.q_posterior(t, x_t, x0_pred, y_wavelet)

        pred_img = self._inverse_wavelet(x0_pred, out_size)

        adv_loss = self.adversarial_loss(
            self.discriminator(x_prev_fake, t, x_t.detach()),
            is_real=True
        )
        c = self.base_channels

        # ll_rec_loss = F.l1_loss(x0_pred[:, :c], x0_wavelet[:, :c], reduction="sum")
        # hf_rec_loss = F.l1_loss(x0_pred[:, c:], x0_wavelet[:, c:], reduction="sum")
        ll_rec_loss = F.l1_loss(x0_pred[:, :c], x0_wavelet[:, :c], reduction="mean")
        hf_rec_loss = F.l1_loss(x0_pred[:, c:], x0_wavelet[:, c:], reduction="mean")

        rec_loss = (
            self.lambda_ll_loss * ll_rec_loss
            + self.lambda_hf_loss * hf_rec_loss
        )

        g_loss = (
            self.lambda_adv_loss * adv_loss
            + self.lambda_rec_loss * rec_loss
        )

        
        self.manual_backward(g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # =========================
        # 3. Logging
        # =========================
        self.log("d_loss/real", d_real_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("d_loss/fake", d_fake_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("d_loss/gp", gp_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("d_loss/total", d_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("d_acc", d_acc.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("d_acc/real", d_real_acc.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("d_acc/fake", d_fake_acc.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("d_logit/real_mean", disc_real.detach().mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("d_logit/fake_mean", disc_fake.detach().mean(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log("g_loss/adv", adv_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/ll_rec", ll_rec_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/hf_rec", hf_rec_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/adv_weighted", (self.lambda_adv_loss * adv_loss).detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/rec_weighted", (self.lambda_rec_loss * rec_loss).detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/rec", rec_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/total", g_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    @torch.inference_mode()
    def _sample_image(self, batch) -> Tensor:
        x0, _, _ = batch
        out = self.wavelet_transform(batch)
        y_wavelet = out["y_wavelet"]
        pred_wavelet = self.diffusion.sample_wavelet_x0(y_wavelet, self._predict_wavelet_x0)
        pred_img = self._inverse_wavelet(pred_wavelet, x0.shape[-2:])
        return torch.clamp(pred_img, -1.0, 1.0)
    def _to01_for_check(self, x):
        x = x.detach().float()
        if x.min() < -0.1:
            x = (x + 1.0) / 2.0
        return x.clamp(0, 1)
    # def validation_step(self, batch, batch_idx):
    #     x0, _, _ = batch
    #     x0_pred = self._sample_image(batch)
    #     loss = F.mse_loss(x0_pred, x0)
    #     metrics = compute_metrics(x0, x0_pred)
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
    #     self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
    #     self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
    #     print("x0", x0.min().item(), x0.max().item(), x0.shape)
    #     print("x0_wavelet", x0_wavelet.min().item(), x0_wavelet.max().item(), x0_wavelet.shape)
    #     print("y_wavelet", y_wavelet.min().item(), y_wavelet.max().item(), y_wavelet.shape)
    #     print("x_t", x_t.min().item(), x_t.max().item(), x_t.shape)
    #     print("x0_pred", x0_pred.min().item(), x0_pred.max().item(), x0_pred.shape)
    #     print("pred_img", pred_img.min().item(), pred_img.max().item(), pred_img.shape)
    #     path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
    #     save_image_pair(x0, x0_pred, path)
    def validation_step(self, batch, batch_idx):
        x0, _, _ = batch

        out = self.wavelet_transform(batch)
        x0_wavelet = out["x0_wavelet"]
        y_wavelet = out["y_wavelet"]

        pred_wavelet = self.diffusion.sample_wavelet_x0(
            y_wavelet,
            self._predict_wavelet_x0,
        )

        pred_img = self._inverse_wavelet(pred_wavelet, x0.shape[-2:])
        x0_pred = torch.clamp(pred_img, -1.0, 1.0)

        loss = F.mse_loss(x0_pred, x0)
        metrics = compute_metrics(x0, x0_pred)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psnr", metrics["psnr_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"].mean(), on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            with torch.no_grad():
                t_dbg = torch.full(
                    (x0_wavelet.shape[0],),
                    self.n_steps,
                    device=x0_wavelet.device,
                    dtype=torch.long,
                )

                _, x_t_dbg = self.diffusion.q_sample_mixed_pair(
                    t_dbg,
                    x0_wavelet,
                    y_wavelet,
                )

                print("\n[Validation range check]")
                print("x0", x0.min().item(), x0.max().item(), x0.shape)
                print("x0_wavelet", x0_wavelet.min().item(), x0_wavelet.max().item(), x0_wavelet.shape)
                print("y_wavelet", y_wavelet.min().item(), y_wavelet.max().item(), y_wavelet.shape)
                print("x_t_dbg", x_t_dbg.min().item(), x_t_dbg.max().item(), x_t_dbg.shape)
                print("pred_wavelet", pred_wavelet.min().item(), pred_wavelet.max().item(), pred_wavelet.shape)
                print("pred_img_raw", pred_img.min().item(), pred_img.max().item(), pred_img.shape)
                print("x0_pred_clamped", x0_pred.min().item(), x0_pred.max().item(), x0_pred.shape)
                print("x0_wavelet abs mean", x0_wavelet.abs().mean().item())
                print("pred_wavelet abs mean", pred_wavelet.abs().mean().item())
                print("pred_img abs mean", pred_img.abs().mean().item())

            path = os.path.join(
                self.logger.log_dir,
                "val_samples",
                f"epoch_{self.current_epoch}.png",
            )
            save_image_pair(x0, x0_pred, path)
    def on_test_start(self):
        self.test_samples = []
        self.mask = None
        self.subject_ids = None
        if self.eval_mask:
            self.mask = self.trainer.datamodule.test_dataset._load_data("mask")
        if self.eval_subject:
            self.subject_ids = self.trainer.datamodule.test_dataset.subject_ids

    def test_step(self, batch, batch_idx):
        x0, _, slice_idx = batch
        x0_pred = self._sample_image(batch)
        all_pred = self.all_gather(x0_pred)
        slice_indices = self.all_gather(slice_idx)
        if self.global_rank == 0:
            h, w = x0.shape[-2:]
            self.test_samples.extend(
                list(zip(slice_indices.flatten().tolist(), all_pred.reshape(-1, h, w).cpu().numpy()))
            )

    def on_test_end(self):
        if self.global_rank != 0:
            return
        self.test_samples.sort(key=lambda x: x[0])
        pred = np.array([x[1] for x in self.test_samples])
        slice_indices = np.array([x[0] for x in self.test_samples])
        _, locs = np.unique(slice_indices, return_index=True)
        pred = pred[locs]

        dataset = self.trainer.datamodule.test_dataset
        source = dataset.source
        target = dataset.target
        save_preds(pred, os.path.join(self.logger.log_dir, "test_samples", "pred.npy"))

        metrics = compute_metrics(
            gt_images=target,
            pred_images=pred,
            mask=self.mask,
            subject_ids=self.subject_ids,
            report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt"),
        )
        print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
        print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")

        indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
        save_eval_images(
            source_images=source[indices],
            target_images=target[indices],
            pred_images=pred[indices],
            psnrs=metrics["psnrs"][indices],
            ssims=metrics["ssims"][indices],
            save_path=os.path.join(self.logger.log_dir, "test_samples"),
        )

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for scheduler in schedulers:
            scheduler.step()

    def configure_optimizers(self):
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr_g, betas=self.optim_betas)
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas)
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_d = CosineAnnealingLR(optimizer_d, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        if "test" in self.parser.args and "CSVLogger" in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"
        super().instantiate_classes()


def cli_main():
    _LightningCLI(
        BridgeRunner,
        DataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    cli_main()
