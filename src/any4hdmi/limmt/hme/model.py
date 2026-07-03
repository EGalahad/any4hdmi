from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PeriodicAutoencoder(nn.Module):
    def __init__(
        self,
        inp_ch: int,
        latent_ch: int,
        win_len: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        win_sec: float = 4.0,
    ) -> None:
        super().__init__()
        self.inp_ch = int(inp_ch)
        self.latent_ch = int(latent_ch)
        self.win_len = int(win_len)
        if self.win_len % 2 == 0:
            raise ValueError("win_len must be odd")
        padding = self.win_len // 2

        self.register_buffer("time_vec", torch.linspace(-win_sec / 2, win_sec / 2, self.win_len), persistent=False)
        self.register_buffer("freqs", torch.fft.rfftfreq(self.win_len)[1:] * self.win_len / win_sec, persistent=False)
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi, dtype=torch.float32), persistent=False)

        enc_layers: list[nn.Module] = []
        enc_channels = (self.inp_ch, *hidden_dims, self.latent_ch)
        for in_ch, out_ch in zip(enc_channels[:-1], enc_channels[1:], strict=True):
            enc_layers.extend([nn.Conv1d(in_ch, out_ch, self.win_len, padding=padding), nn.BatchNorm1d(out_ch), nn.ELU()])
        self.encoder = nn.Sequential(*enc_layers)

        self.phase_encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.win_len, 2), nn.BatchNorm1d(2)) for _ in range(self.latent_ch)]
        )

        dec_layers: list[nn.Module] = []
        dec_channels = (self.latent_ch, *hidden_dims)
        for in_ch, out_ch in zip(dec_channels[:-1], dec_channels[1:], strict=True):
            dec_layers.extend([nn.Conv1d(in_ch, out_ch, self.win_len, padding=padding), nn.BatchNorm1d(out_ch), nn.ELU()])
        last_ch = dec_channels[-1]
        dec_layers.append(nn.Conv1d(last_ch, self.inp_ch, self.win_len, padding=padding))
        self.decoder = nn.Sequential(*dec_layers)

    def _compute_fft_params(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rfft = torch.fft.rfft(latent, dim=-1)
        spectrum = torch.abs(rfft[:, :, 1:])
        power = spectrum.square()
        pow_sum = power.sum(dim=-1).add(1e-8)
        freq = (self.freqs * power).sum(dim=-1) / pow_sum
        amp = 2 * torch.sqrt(pow_sum) / self.win_len
        offset = rfft.real[:, :, 0] / self.win_len
        return amp.unsqueeze(-1), freq.unsqueeze(-1), offset.unsqueeze(-1)

    def encode(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(inp)
        amp, freq, offset = self._compute_fft_params(latent)
        shifts = []
        for channel, phase_encoder in enumerate(self.phase_encoders):
            z_shift = phase_encoder(latent[:, channel])
            shifts.append(torch.atan2(z_shift[..., 1], z_shift[..., 0]) / self.two_pi)
        shift = torch.stack(shifts, dim=1).unsqueeze(-1)
        return {"latent": latent, "amp": amp, "freq": freq, "offset": offset, "shift": shift}

    def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded_params = self.encode(inp)
        recon_latent = (
            encoded_params["amp"]
            * torch.sin(self.two_pi * (encoded_params["freq"] * self.time_vec + encoded_params["shift"]))
            + encoded_params["offset"]
        )
        reconstruction = self.decoder(recon_latent)
        return {"pred": reconstruction, "loss": F.mse_loss(reconstruction, inp)}
