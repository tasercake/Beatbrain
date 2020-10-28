import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram


class InvertibleMelSpectrogram(Spectrogram.MelSpectrogram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable_mel = kwargs.get("trainable_mel", False)
        self.trainable_stft = kwargs.get("trainable_stft", False)
        self.verbose = kwargs.get("verbose", False)

    def to_stft(self, melspec, max_steps=1000, loss_threshold=1e-8, grad_threshold=1e-7, random_start=False, sgd_kwargs=None, eps=1e-12, return_extras=False, verbose=None):
        """
        Best-attempt spectrogram inversion
        """
        def loss_fn(pred, target):
            pred = pred.unsqueeze(1) if pred.ndim == 3 else pred
            target = target.unsqueeze(1) if target.ndim == 3 else target

            loss = (pred - target).pow(2).sum(-2).mean()
            return loss

        verbose = verbose or self.verbose
        # SGD arguments
        default_sgd_kwargs = dict(lr=1e3, momentum=0.9)
        if sgd_kwargs:
            default_sgd_kwargs.update(sgd_kwargs)
        sgd_kwargs = default_sgd_kwargs

        mel_basis = self.mel_basis.detach()
        shape = melspec.shape
        batch_size, n_mels, time = shape[0], shape[-2], shape[-1]
        _, n_freq = mel_basis.shape
        melspec = melspec.detach().view(-1, n_mels, time)
        if random_start:
            pred_stft_shape = (batch_size, n_freq, time)
            pred_stft = torch.zeros(*pred_stft_shape, dtype=torch.float32, device=mel_basis.device).normal_().clamp_(eps)
        else:
            pred_stft = (torch.pinverse(mel_basis) @ melspec).clamp(eps)
        pred_stft = nn.Parameter(pred_stft, requires_grad=True)

        sgd_kwargs["lr"] = sgd_kwargs["lr"] * batch_size
        optimizer = torch.optim.SGD([pred_stft], **sgd_kwargs)

        losses = []
        for i in range(max_steps):
            optimizer.zero_grad()
            pred_mel = mel_basis @ pred_stft
            loss = loss_fn(pred_mel, melspec)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Check conditions
            if not loss.isfinite():
                raise OverflowError("Overflow encountered in Mel -> STFT optimization")
            if loss_threshold and loss < loss_threshold:
                if verbose:
                    print(f"Target error of {loss_threshold} reached. Stopping optimization.")
                break
            if grad_threshold and pred_stft.grad.max() < grad_threshold:
                if verbose:
                    print(f"Target max gradient of {grad_threshold} reached. Stopping optimization.")
                break

        pred_stft = pred_stft.detach().clamp(eps) ** 0.5
        if return_extras:
            return pred_stft, pred_mel.detach(), losses
        return pred_stft.view((*shape[:-2], freq, time))


def psnr(pred, target, target_top=True, top=1e4):
    """
    Peak Signal-to-Noise Ratio (but not really)
    Since spectrograms values are unbounded, this function uses `max(target)` as the maximum possible value by default.
    """
    if target_top:
        top = target.max()
    return 20 * torch.log10(top / torch.sqrt(F.mse_loss(pred, target)))
