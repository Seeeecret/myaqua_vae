import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class WatermarkSystem(nn.Module):
    def __init__(self, latent_dim=256, watermark_dim=121):
        super().__init__()
        # 正交投影矩阵
        self.P = nn.Parameter(torch.randn(latent_dim, watermark_dim)).cuda()
        nn.init.orthogonal_(self.P)
        self.watermark_dim = watermark_dim

        # 动态强度控制器
        self.alpha_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Arnold变换参数
        self.register_buffer('arnold_matrix', torch.tensor([[1, 1], [1, 2]], dtype=torch.float32))
        self.register_buffer('inv_arnold', torch.tensor([[2, -1], [-1, 1]], dtype=torch.float32))

    def arnold_transform(self, w, iterations=3, reverse=False):
        """处理任意正方形水印"""
        B, D = w.shape
        N = int(D ** 0.5)
        assert N * N == D, "输入维度必须是平方数"

        w = w.view(B, N, N)  # [B, N, N]

        for _ in range(iterations):
            # 生成坐标网格（设备敏感）
            coords = torch.stack(torch.meshgrid(
                torch.arange(N, device=w.device),
                torch.arange(N, device=w.device)
            ), -1).float()

            # 变换矩阵选择
            matrix = self.inv_arnold if reverse else self.arnold_matrix

            # 确保矩阵和坐标在相同的设备上
            matrix = matrix.to(w.device)

            # 计算新坐标
            new_coords = (matrix @ coords.unsqueeze(-1)).squeeze() % N

            # 归一化到[-1,1]范围
            grid = torch.stack([
                new_coords[..., 1] / (N - 1) * 2 - 1,  # 高度维
                new_coords[..., 0] / (N - 1) * 2 - 1  # 宽度维
            ], dim=-1)

            # 双线性插值
            w = F.grid_sample(
                w.unsqueeze(1),  # [B,1,N,N]
                grid.unsqueeze(0).repeat(B, 1, 1, 1),  # [B,N,N,2]
                mode='bilinear',
                align_corners=True
            ).squeeze(1)  # [B,N,N]

        return w.view(B, N * N)  # 展平返回


    def forward(self, z, watermark=None):
        """水印注入"""
        if watermark is None:
            N = int(self.watermark_dim ** 0.5)
            w = torch.randint(0, 2, (z.size(0), N, N), device=z.device).float().cuda()
            w = w.view(z.size(0), -1)  # 展平为[B, 121]
            w = self.arnold_transform(w)
        else:
            w = watermark

        # 确保 alpha_net 和 self.P 在与 z 相同的设备上
        self.alpha_net.to(z.device)
        self.P = self.P.to(z.device)  # 确保 P 在正确的设备上

        alpha = self.alpha_net(z) * 0.1

        tmp = (self.P @ w.unsqueeze(-1)).squeeze()

        delta_z = alpha * tmp

        # 频域约束
        z_fft = torch.fft.rfft(z, dim=1)
        mask = (z_fft.abs() > 0.1).float()
        delta_z = torch.fft.irfft(torch.fft.rfft(delta_z, dim=1) * mask, dim=1)

        return z + delta_z, w

    def detect_from_lora(self, lora_params, vae_model):
        """从LoRA参数中检测水印"""
        # 编码到潜在空间
        with torch.no_grad():
            mu, log_var = vae_model.encode(lora_params)
            z_w = vae_model.reparameterize(mu, log_var)

        # 水印提取
        w_pred = torch.pinverse(self.P) @ z_w.T
        w_pred = torch.sigmoid(w_pred.T)
        w_pred = self.arnold_transform(w_pred, iterations=3, reverse=True)
        return (w_pred > 0.5).float()


    # def arnold_transform(self, w, iterations=3, reverse=False):
    #     """改进的Arnold变换，支持正/逆变换"""
    #     N = int(w.shape[-1] ** 0.5)
    #     w = w.view(-1, N, N)
    #
    #     for _ in range(iterations):
    #         coords = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N)), -1).float()
    #         if reverse:
    #             new_coords = (self.inv_arnold @ (coords.unsqueeze(-1) - coords)).squeeze() % N
    #         else:
    #             new_coords = (self.arnold_matrix @ coords.unsqueeze(-1)).squeeze() % N
    #
    #         # 双线性插值保持可逆性
    #         x = new_coords[..., 0].clamp(0, N - 1)
    #         y = new_coords[..., 1].clamp(0, N - 1)
    #
    #         w = F.grid_sample(
    #             w.unsqueeze(1),
    #             torch.stack([y / (N - 1) * 2 - 1, x / (N - 1) * 2 - 1], -1),
    #             mode='bilinear',
    #             align_corners=True
    #         ).squeeze(1)
    #
    #     return w.view(-1, N * N)

# class ImageWatermarkDetector(nn.Module):
#     """专注于从生成图像中提取水印的专用网络"""
#
#     def __init__(self, watermark_dim=128):
#         super().__init__()
#         # 多尺度特征提取
#         self.feature_net = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 128x128
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
#             nn.AdaptiveAvgPool2d(16)  # 统一到16x16
#         )
#
#         # 注意力机制聚焦水印区域
#         self.attention = nn.Sequential(
#             nn.Conv2d(128, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # 水印解码
#         self.decoder = nn.Sequential(
#             nn.Linear(128 * 16 * 16, 512),
#             nn.ReLU(),
#             nn.Linear(512, watermark_dim),
#             nn.Sigmoid()
#         )
#

class ImageWatermarkDetector2(nn.Module):
    """从生成图像中检测水印的神经网络"""

    def __init__(self, watermark_dim=128):
        super().__init__()

        # 小波特征提取器
        self.wavelet_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 水印解码
        self.decoder = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, watermark_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x: [B, 3, 256, 256]
        feat = self.wavelet_conv(x)  # [B, 64, 64, 64]

        # 小波变换增强特征
        coeffs = pywt.dwt2(feat.detach().cpu().numpy(), 'haar')
        cA, (cH, cV, cD) = coeffs
        cD = torch.tensor(cD, device=x.device)
        feat = feat * cD.unsqueeze(1)

        # 注意力聚焦
        attn = self.attention(feat)
        feat = feat * attn

        # 解码水印
        feat = feat.view(feat.size(0), -1)
        return self.decoder(feat)


class WatermarkedVAE(nn.Module):
    def __init__(self, base_vae, watermark_system):
        super().__init__()
        self.encoder = base_vae.encoder
        self.decoder = base_vae.decoder
        self.watermark = watermark_system

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z_w, w = self.watermark(z)
        return self.decode(z_w), w


#
class ImageWatermarkDetector3(nn.Module):
    """专注于从生成图像中提取水印的专用网络"""

    def __init__(self, watermark_dim=121):
        super().__init__()
        # 多尺度特征提取
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 128x128
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.AdaptiveAvgPool2d(16)  # 统一到16x16
        )

        # 注意力机制聚焦水印区域
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 水印解码
        self.decoder = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, watermark_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 目标输入x: [B, 3, 256, 256]
        # 如果输入数据的shape是(B, 512, 512, 3)或(B, 256, 256, 3)，则需要进行转换
        # 处理输入shape为(B, 512, 512, 3)的情况
        # 获取模型的当前设备
        device = next(self.parameters()).device

        # 确保输入x在相同的设备上
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x = x.to(device)  # 将输入x移动到与模型相同的设备

        if x.shape[1] == 512:
            x = x.permute(0, 3, 1, 2)  # 从 [B, 512, 512, 3] 转换为 [B, 3, 512, 512]
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)  # 缩放为 [B, 3, 256, 256]

        # 处理输入shape为(B, 256, 256, 3)的情况
        elif x.shape[1] == 256:
            x = x.permute(0, 3, 1, 2)  # 从 [B, 256, 256, 3] 转换为 [B, 3, 256, 256]


        features = self.feature_net(x)  # [B, 128, 16, 16]

        # 注意力权重
        attn = self.attention(features)  # [B, 1, 16, 16]
        weighted_feat = features * attn  # 聚焦关键区域

        # 解码水印
        flattened = weighted_feat.view(x.size(0), -1)
        return self.decoder(flattened)  # [B, watermark_dim]