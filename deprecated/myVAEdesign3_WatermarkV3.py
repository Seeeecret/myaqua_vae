# import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../watermark/")
sys.path.append("../")

from watermark.V4.util import detect_watermark



# 水印加在Z上的版本
class OneDimVAEWatermark(nn.Module):
    def __init__(self, input_length, latent_dim=256, kernel_size=7, divide_slice_length=1024, kld_weight=0.02,
                 target_fpr=1e-6, lambda_w=1.0, n_iters=50,**kwargs):
        super(OneDimVAEWatermark, self).__init__()

        self.target_fpr = target_fpr
        self.lambda_w = lambda_w
        self.loss_recon = None
        # 初始化密钥（随机单位向量）和角度阈值
        self.register_buffer("carrier", torch.randn(1, latent_dim))
        self.carrier = F.normalize(self.carrier, dim=1)  # 单位化
        self.angle = self._compute_angle(latent_dim, target_fpr)  # +++ 计算角度θ
        self.n_iters = n_iters  # 优化迭代次数

        d_model = [8, 16, 32, 64, 128, 256, 256, 128, 64, 32]
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
    def detect(self, z):
        """ 直接检测 μ 中的水印 """
        return detect_watermark(
            z,
            self.carrier,
            self.angle,
            return_confidence=True
        )
    def _compute_angle(self, d, fpr):
        # 根据论文公式4计算角度θ（简化实现）
        from scipy.special import betaincinv
        a = 0.5 * (d - 1)
        b = 0.5
        tau = betaincinv(a, b, 1 - fpr)  # 求解Beta分布的逆函数
        return np.arccos(np.sqrt(tau))
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
        # Check input dimensions
        # if input.dim() == 2:  # [batch_size, sequence_length]
        #     input = input[:, None, :]  # Add channel dimension
        # elif input.dim() == 3:  # [batch_size, 1, sequence_length]
        #     pass  # Input shape is already correct
        # 填充序列
        input = self.pad_sequence(input)  # [B, 1, adjusted_input_length]

        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

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

    def watermark(self, z):
        # 计算水印并添加到 z 中
        # 保留原始 z 的副本
        # z_original = z.clone().detach().requires_grad_(True)

        # 直接优化 z
        z_watermarked = self._watermark_optim(z, n_iters=100, lr=0.01)

        return z_watermarked, z

    def encode_decode(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)

        # 添加水印
        z_watermarked, z_original = self.watermark(z)

        # 约束优化后的 z 不要偏离原始 z 太远
        loss_w = F.mse_loss(z_watermarked, z_original)
        # 返回约束距离的损失loss_w，在forward函数里将loss_w与recon_loss和KLD_loss相加得到总损失再backward
        loss_w *= self.lambda_w

        recons = self.decode(z_watermarked)
        return recons, input, mu, log_var, loss_w

    def _watermark_optim(self, z, n_iters=50, lr=0.01):
        # 创建 z_opt 并进行优化，但保持计算图连通
        z_opt = z.clone().detach().requires_grad_(True)  # 将 z 转换为叶子张量
        # z_opt.requires_grad_(True)
        optimizer = torch.optim.Adam([z_opt], lr=lr)

        cos_theta = np.cos(self.angle)
        for _ in range(n_iters):
            # 计算超锥体损失（论文公式5）
            dot_product = z_opt @ self.carrier.T
            norm = torch.norm(z_opt, dim=1, keepdim=True)
            loss_w = torch.mean(torch.relu(norm * cos_theta - dot_product))

            # 总损失 = 水印损失
            loss = self.lambda_w * loss_w
            if hasattr(self, 'loss_recon') and self.loss_recon is not None:
                loss += self.loss_recon

            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        z_final = z_opt.detach() + 0.0 * z  # 保持 z 的计算图，但替换为 z_opt

        return z_final
    def sample(self, batch=1):
        z = torch.randn((batch, self.d_latent), device=self.device, dtype=torch.float32)
        recons = self.decode(z)
        return recons

    def forward(self, x, **kwargs):
        recons, input, mu, log_var,loss_w = self.encode_decode(input=x, **kwargs)
        padded_x = self.pad_sequence(x)
        if recons is not None and recons.dim() == 3:
            recons = recons.squeeze(1)
        if padded_x.dim() == 3:
            padded_x = padded_x.squeeze(1)

        recons_loss = F.mse_loss(input=recons, target=padded_x, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss + loss_w

        return loss, recons_loss, kld_loss, loss_w

    @property
    def device(self):
        return next(self.parameters()).device
