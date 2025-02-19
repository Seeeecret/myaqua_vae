import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkEncoder(nn.Module):
    def __init__(self, watermark_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(watermark_dim, latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.Tanh()  # 限制水印强度
        )

    def forward(self, w):
        return self.encoder(w)


class WatermarkedOneDimVAE(nn.Module):
    def __init__(self, input_length, latent_dim=256, kernel_size=7,
                 divide_slice_length=1024, kld_weight=0.02,
                 watermark_dim=64, watermark_strength=0.1, **kwargs):
        super(WatermarkedOneDimVAE, self).__init__()
        d_model = [8, 16, 32, 64, 128, 256, 256, 128, 64, 32]
        self.d_model = d_model
        self.d_latent = latent_dim
        self.kld_weight = kld_weight
        self.watermark_strength = watermark_strength
        self.divide_slice_length = divide_slice_length
        self.initial_input_length = input_length

        # 水印相关组件
        self.watermark_encoder = WatermarkEncoder(watermark_dim, latent_dim)
        self.watermark_detector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim // 2, watermark_dim)
        )

        input_length = (input_length // divide_slice_length + 1) * divide_slice_length \
            if input_length % divide_slice_length != 0 else input_length
        assert input_length % int(2 ** len(d_model)) == 0

        self.adjusted_input_length = input_length
        self.last_length = input_length // int(2 ** len(d_model))

        # Encoder
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, 2, kernel_size // 2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Linear(self.last_length * d_model[-1], latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # Decoder
        modules = []
        self.to_decode = nn.Linear(latent_dim, self.last_length * d_model[-1])
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i + 1], kernel_size, 2, kernel_size // 2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.ELU(),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, 2, kernel_size // 2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, 1, kernel_size // 2),
        )

    def pad_sequence(self, input_seq):
        batch_size, channels, seq_length = input_seq.size()
        if seq_length < self.adjusted_input_length:
            padding_size = self.adjusted_input_length - seq_length
            input_seq = F.pad(input_seq, (0, padding_size), "constant", 0)
        elif seq_length > self.adjusted_input_length:
            input_seq = input_seq[:, :, :self.adjusted_input_length]
        return input_seq

    def encode(self, input, watermark=None, **kwargs):
        if input.dim() == 2:
            input = input[:, None, :]
        input = self.pad_sequence(input)

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        if watermark is not None:
            # 编码水印
            w_encoded = self.watermark_encoder(watermark)
            # 使用Gram-Schmidt正交化确保水印和主要内容正交
            w_encoded = w_encoded - torch.sum(w_encoded * mu, dim=1, keepdim=True) * mu / (
                        torch.sum(mu * mu, dim=1, keepdim=True) + 1e-6)
            # 加入水印
            mu = mu + self.watermark_strength * w_encoded

        return mu, log_var

    def decode(self, z, **kwargs):
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[0], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result[:, 0, :]

    def detect_watermark(self, z):
        return self.watermark_detector(z)

    def reparameterize(self, mu, log_var, **kwargs):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if kwargs.get("manual_std") is not None:
            std = kwargs.get("manual_std")
        return eps * std + mu

    def encode_decode(self, input, watermark=None, **kwargs):
        mu, log_var = self.encode(input, watermark)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var, z

    def forward(self, x, watermark=None, **kwargs):
        recons, input, mu, log_var, z = self.encode_decode(input=x, watermark=watermark, **kwargs)
        padded_x = self.pad_sequence(x)

        if recons is not None and recons.dim() == 3:
            recons = recons.squeeze(1)
        if padded_x.dim() == 3:
            padded_x = padded_x.squeeze(1)

        # 重建损失
        recons_loss = F.mse_loss(recons, padded_x, reduction='mean')

        # KLD损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # 水印损失
        watermark_loss = 0
        if watermark is not None:
            detected_watermark = self.detect_watermark(z)
            watermark_loss = F.mse_loss(detected_watermark, watermark)

        # 总损失
        loss = recons_loss + self.kld_weight * kld_loss + 0.1 * watermark_loss

        return loss, recons_loss, kld_loss, watermark_loss

    @property
    def device(self):
        return next(self.parameters()).device