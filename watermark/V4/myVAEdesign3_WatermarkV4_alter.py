# import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from watermark.V4.util import detect_watermark_in_mu


# 水印加在μ上的版本，然后水印检测也在μ上
class WatermarkLayer(nn.Module):
    def __init__(self, latent_dim, target_fpr=1e-5, lambda_w=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_w = lambda_w

        # 初始化密钥（随机单位向量）
        self.register_buffer("carrier", F.normalize(torch.randn(1, latent_dim), dim=1))


        # 计算超锥体角度θ（根据目标FPR）
        self.angle = self._compute_angle(target_fpr)

    def _compute_angle(self, fpr):
        # 根据论文公式4计算角度θ（简化实现）
        from scipy.special import betaincinv
        a = 0.5 * (self.latent_dim - 1)
        b = 0.5
        tau = betaincinv(a, b, 1 - fpr)
        return np.arccos(np.sqrt(tau))

    def forward(self, mu, log_var):
        """
        输入:
            mu: 编码器输出的均值向量 [B, latent_dim]
            log_var: 编码器输出的对数方差 [B, latent_dim]
        输出:
            mu_wm: 水印优化后的均值向量 [B, latent_dim]
            log_var: 未修改的对数方差（保持随机性）
        """
        # 水印优化目标：强制 mu_wm 满足超锥体条件
        mu_wm,loss_w = self._optimize_mu(mu)
        return mu_wm, log_var, loss_w
    def detect_in_mu(self, mu):
        """ 直接检测 μ 中的水印 """
        return detect_watermark_in_mu(
            mu,
            self.carrier,
            self.angle,
            return_confidence=True
        )
    def _optimize_mu(self, mu):
        """ 对 mu 进行梯度下降优化 """
        mu_opt = mu.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu_opt], lr=0.01)

        cos_theta = np.cos(self.angle)
        for _ in range(50):  # 优化迭代次数
            dot_product = mu_opt @ self.carrier.T
            norm_mu = torch.norm(mu_opt, dim=1, keepdim=True)
            loss_w = torch.mean(torch.relu(norm_mu * cos_theta - dot_product))
            loss_w = self.lambda_w * loss_w

            optimizer.zero_grad()
            loss_w.backward()
            optimizer.step()

        # 将优化后的 mu 反向绑定到原始 mu（保留梯度）
        mu_wm = mu_opt.detach() + 0.0 * mu  # 保持计算图连通
        return mu_wm, loss_w

class OneDimVAEWatermark(nn.Module):
    def __init__(self, input_length, latent_dim=256, kernel_size=7, divide_slice_length=4096, kld_weight=0.02,
                 target_fpr=1e-6, lambda_w=1.0, **kwargs):
        super(OneDimVAEWatermark, self).__init__()

        self.watermark_layer = WatermarkLayer(
            latent_dim,
            target_fpr=target_fpr,  # 使用传入的 target_fpr
            lambda_w=lambda_w  # 使用传入的 lambda_w
        )
        d_model = [8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8]
        self.d_model = d_model
        self.d_latent = latent_dim
        self.kld_weight = kld_weight
        self.divide_slice_length = divide_slice_length
        self.initial_input_length = input_length
        # confirm self.last_length
        input_length = (input_length // divide_slice_length + 1) * divide_slice_length \
            if input_length % divide_slice_length != 0 else input_length
        assert input_length % int(2 ** len(d_model)) == 0, \
            f"Please set divide_slice_length to {int(2 ** len(d_model))}."

        self.adjusted_input_length = input_length
        self.last_length = input_length // int(2 ** len(d_model))

        # Build Encoder
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

        # Build Decoder
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
        """
        在序列末尾添加零以调整长度到 self.adjusted_input_length。
        """
        batch_size, channels, seq_length = input_seq.size()
        if seq_length < self.adjusted_input_length:
            padding_size = self.adjusted_input_length - seq_length
            # 在最后一个维度上填充
            input_seq = F.pad(input_seq, (0, padding_size), "constant", 0)
        elif seq_length > self.adjusted_input_length:
            # 截断多余的部分
            input_seq = input_seq[:, :self.adjusted_input_length]
        return input_seq

    def encode(self, input, **kwargs):
        # 填充序列
        input = self.pad_sequence(input)  # [B, 1, adjusted_input_length]

        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def watermark_encode(self, mu, log_var):
        mu_wm, log_var,loss_w = self.watermark_layer(mu, log_var)
        return mu_wm, log_var,loss_w

    def decode(self, z, **kwargs):
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[0], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result[:, 0, :]

    def reparameterize(self, mu, log_var, **kwargs):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if kwargs.get("manual_std") is not None:
            std = kwargs.get("manual_std")
        return eps * std + mu

    def encode_decode(self, input, apply_watermark=False, **kwargs):
        mu, log_var = self.encode(input)
        if apply_watermark:
            mu, log_var,loss_w = self.watermark_encode(mu, log_var)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var, loss_w

    def sample(self, batch=1):
        z = torch.randn((batch, self.d_latent), device=self.device, dtype=torch.float32)
        recons = self.decode(z)
        return recons

    def forward(self, x, apply_watermark=False, **kwargs):
        recons, input, mu, log_var, loss_w = self.encode_decode(input=x, apply_watermark=True, **kwargs)
        padded_x = self.pad_sequence(x)
        if recons is not None and recons.dim() == 3:
            recons = recons.squeeze(1)
        if padded_x.dim() == 3:
            padded_x = padded_x.squeeze(1)
        recons_loss = F.mse_loss(input=recons, target=padded_x, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss

        return loss, recons_loss, kld_loss, loss_w

    @property
    def device(self):
        return next(self.parameters()).device
