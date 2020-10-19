import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram


class InvertibleMelSpectrogram(Spectrogram.MelSpectrogram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable_mel = kwargs.get("trainable_mel")
        self.trainable_stft = kwargs.get("trainable_stft")

    def to_stft(self, melspec, max_steps=1000, loss_threshold=1e-12, psnr_threshold=200, change_threshold=1e-16, sgd_kwargs=None, lr_scheduler_kwargs=None, eps=1e-12, return_extras=False,):
        """
        Best-attempt spectrogram inversion
        """
        def loss_fn(pred, target):
            pred = pred.unsqueeze(1) if pred.ndim == 3 else pred
            target = target.unsqueeze(1) if target.ndim == 3 else target

            loss = (pred - target).pow(2).mean()
            return loss

        # SGD arguments
        default_sgd_kwargs = dict(lr=1e6, momentum=0.9)
        if sgd_kwargs:
            default_sgd_kwargs.update(sgd_kwargs)
        sgd_kwargs = default_sgd_kwargs
        # ReduceLROnPlateau arguments
        default_scheduler_kwargs = dict(factor=0.1, patience=500, threshold=1e-6, min_lr=1e2, verbose=True)
        if lr_scheduler_kwargs:
            default_scheduler_kwargs.update(lr_scheduler_kwargs)
        lr_scheduler_kwargs = default_scheduler_kwargs

        melspec = melspec.detach()
        mel_basis = self.mel_basis.detach()
        pred_stft = (torch.pinverse(mel_basis) @ melspec).clamp(eps)
#         pred_stft_shape = (melspec.shape[0], mel_basis.shape[-1], melspec.shape[-1])
#         pred_stft = torch.zeros(*pred_stft_shape, dtype=torch.float32, device=DEVICE).normal_().clamp_(eps)
        pred_stft = nn.Parameter(pred_stft, requires_grad=True)

        optimizer = torch.optim.SGD([pred_stft], **sgd_kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)

        losses = []
        for i in range(max_steps):
            optimizer.zero_grad()
            pred_mel = mel_basis @ pred_stft
            loss = loss_fn(pred_mel, melspec)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Check conditions
            if psnr(pred_mel, melspec) >= psnr_threshold:
                if self.verbose:
                    print(f"Target Mel PSNR of {psnr_threshold} reached. Stopping optimization.")
                break
            if loss <= loss_threshold:
                if self.verbose:
                    print(f"Target error of {loss_threshold} reached. Stopping optimization.")
                break
            if i > 1 and abs(losses[-2] - losses[-1]) <= change_threshold:
                if self.verbose:
                    print(f"Target loss change of {change_threshold} reached. Stopping optimization.")
                break

        pred_stft = pred_stft.detach().clamp(eps) ** 0.5
        if return_extras:
            return pred_stft, pred_mel.detach(), losses
        return pred_stft
