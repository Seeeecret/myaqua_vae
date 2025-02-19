import torch
import torch.nn as nn
import torch.nn.functional as F

# 水印模块定义
class WatermarkSystem(nn.Module):
    def __init__(self, latent_dim=256, watermark_dim=128):
        super().__init__()
        # 正交投影矩阵
        self.P = nn.Parameter(torch.randn(latent_dim, watermark_dim))
        nn.init.orthogonal_(self.P)  # 正交初始化

        # 动态强度控制器
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出范围[0,1]
        )

        # 水印加密参数
        self.register_buffer('arnold_matrix', torch.tensor([[1, 1], [1, 2]], dtype=torch.float32))

    def arnold_transform(self, w, iterations=3):
        """Arnold置乱加密"""
        N = int(w.shape[-1] ** 0.5)
        for _ in range(iterations):
            w = w.view(-1, N, N)
            xy = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N)), -1).float()
            xy = (xy + self.arnold_matrix @ xy.unsqueeze(-1)).squeeze() % N
            w = w[:, xy[..., 0].long(), xy[..., 1].long()]
        return w.view(-1, N * N)

    def forward(self, z, watermark=None):
        """
        z: 原始潜在变量 [B, latent_dim]
        watermark: 外部水印 [B, watermark_dim] (若为None则自动生成)
        """
        if watermark is None:
            # 生成基准水印模板
            w = torch.randint(0, 2, (z.size(0), 128), device=z.device).float()
            w = self.arnold_transform(w)  # 加密
        else:
            w = watermark

        # 动态扰动强度
        alpha = self.alpha_net(z) * 0.1  # 最大强度限制在0.1

        # 投影扰动
        delta_z = alpha * (self.P @ w.unsqueeze(-1)).squeeze()

        # 高频约束
        z_fft = torch.fft.rfft(z, dim=1)
        delta_z_fft = torch.fft.rfft(delta_z, dim=1)
        delta_z = torch.fft.irfft(delta_z_fft * (abs(z_fft) > 0.1), dim=1)  # 仅在高频区域添加

        return z + delta_z, w

    def extract_watermark(z_w, vae_model):
        """
        水印提取接口
        z_w: 带水印的潜在变量 [B, latent_dim]
        vae_model: 训练好的水印VAE模型
        """
        # 提取投影矩阵
        P = vae_model.watermark.P.detach()

        # 伪逆解算
        w_pred = torch.pinverse(P) @ z_w.T
        w_pred = torch.sigmoid(w_pred.T)  # 二值化

        # Arnold逆变换
        w_pred = vae_model.watermark.arnold_transform(w_pred, iterations=2)

        return (w_pred > 0.5).float()