# import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F

# 仿照NND的develop分支改造2
class OneDimVAE(nn.Module):
    def __init__(self, latent_dim, input_length, kernel_size=7, kld_weight = 0.005):
        super(OneDimVAE, self).__init__()
        # 定义编码器层数
        # num_layers = 16
        # 定义每一层的通道数
        # d_model = [64] * num_layers  # 或者使用前面示例的通道配置
        d_model = [64, 128, 256, 512, 512, 256, 128, 64]  # 或者

        self.d_model = d_model.copy()
        self.d_latent = latent_dim
        self.last_length = input_length // int(2 ** len(d_model))
        self.kld_weight = kld_weight
        # 构建编码器
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, stride=2, padding=kernel_size//2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Linear(self.last_length * d_model[-1], latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # 构建解码器
        modules = []
        self.to_decode = nn.Linear(latent_dim, self.last_length * d_model[-1])
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i+1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.ELU(),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, stride=1, padding=kernel_size//2),
        )

    # 其余方法保持不变
    def encode(self, input, **kwargs):
        # print(input.shape)
        # assert input.shape == [batch_size, num_parameters]
        # input = input[:, None, :]
        # Check input dimensions
        if input.dim() == 2:  # [batch_size, sequence_length]
            input = input[:, None, :]  # Add channel dimension
        elif input.dim() == 3:  # [batch_size, 1, sequence_length]
            pass  # Input shape is already correct
        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[-1], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        assert result.shape[1] == 1, f"{result.shape}"
        return result[:, 0, :]

    def reparameterize(self, mu, log_var, **kwargs):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_decode(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var

    def sample(self, batch=1):
        z = torch.randn((batch, self.d_latent), device=self.device, dtype=torch.float32)
        recons = self.decode(z)
        return recons

    def forward(self, x, **kwargs):
        recons, input, mu, log_var = self.encode_decode(input=x, **kwargs)
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss +self.kld_weight * kld_loss

        return loss, recons, kld_loss

    @property
    def device(self):
        return next(self.parameters()).device