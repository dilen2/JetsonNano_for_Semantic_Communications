import torch
import torch.nn as nn
import numpy as np

class Channel(nn.Module):
    def __init__(self, chan_type="awgn", device="cuda"):
        super(Channel, self).__init__()
        self.chan_type = chan_type.lower()
        self.device = device

    def gaussian_noise_layer(self, input_layer, std):
        # input_layer: [B, N], std: [B]
        B = input_layer.shape[0]
        if isinstance(std, torch.Tensor):
            std = std.view(B, 1)
        
        noise_real = torch.zeros_like(input_layer, device=self.device).normal_(mean=0.0)
        noise_imag = torch.zeros_like(input_layer, device=self.device).normal_(mean=0.0)
        
        noise = (noise_real + 1j * noise_imag) * std
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std):
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=self.device)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=self.device)
        noise = noise_real + 1j * noise_imag

        h_real = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=self.device)
        h_imag = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=self.device)
        h = torch.sqrt(h_real ** 2 + h_imag ** 2) / np.sqrt(2)

        return input_layer * h + noise

    def complex_normalize(self, x, power=1.0, eps=1e-8):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr + eps)
        return out, pwr

    def complex_forward(self, channel_in, snr_db):
        if self.chan_type == "none":
            return channel_in
        sigma = torch.sqrt(1.0 / (2 * 10 ** (snr_db / 10)))
        if self.chan_type == "awgn":
            return self.gaussian_noise_layer(channel_in, std=sigma)
        elif self.chan_type == "rayleigh":
            return self.rayleigh_noise_layer(channel_in, std=sigma)
        else:
            raise ValueError(f"Unknown channel type: {self.chan_type}")

    def forward(self, input, snr_db, avg_pwr=False):
        if input is None:
            return None
        z_hat = input
        if snr_db is None:
            raise ValueError("snr_db must be provided for Channel.forward")
        if torch.is_tensor(snr_db):
            snr_tensor = snr_db.to(z_hat.device).view(-1)
            if snr_tensor.numel() == 1 and z_hat.size(0) > 1:
                snr_tensor = snr_tensor.expand(z_hat.size(0))
        else:
            snr_tensor = torch.tensor([float(snr_db)], device=z_hat.device)
            if z_hat.size(0) > 1:
                snr_tensor = snr_tensor.expand(z_hat.size(0))
        snr_tensor = snr_tensor.view(-1, 1, 1, 1)

        k = z_hat[0].numel()
        sig_pwr = torch.sum(z_hat.square(), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10.0 ** (snr_tensor / 10.0))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)

        if self.chan_type == 'rayleigh':
            batch = z_hat.size(0)
            hc_real = torch.randn(batch, 1, 1, 1, device=z_hat.device)
            hc_imag = torch.randn(batch, 1, 1, 1, device=z_hat.device)
            hc = torch.sqrt(hc_real ** 2 + hc_imag ** 2) / (2 ** 0.5)
            z_hat = z_hat * hc

        out = z_hat + noise
        return out
