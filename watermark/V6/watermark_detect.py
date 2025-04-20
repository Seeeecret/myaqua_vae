import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFilter
from torchvision.transforms import Compose


class DCTBlockExtractor(nn.Module):
    def __init__(self, freq_bands):
        super().__init__()
        # 创建频率掩码（8×8矩阵，目标频点置1）
        self.freq_bands = freq_bands

        self.register_buffer('freq_mask', self._create_mask(freq_bands))

    def _create_mask(self, bands):
        mask = torch.zeros(8, 8)  # 8×8 DCT块尺寸
        for u, v in bands:  # 遍历配置的频带坐标
            mask[u % 8, v % 8] = 1.0  # 循环处理超出尺寸的情况
        return mask.unsqueeze(0).unsqueeze(0)  # 扩展为[1,1,8,8]

    def forward(self, x):
        # assert x.dim() == 4, f"输入应为4D张量，实际为{x.dim()}D"
        # 自动纠正维度顺序
        # print(x.shape)
        if x.shape[1] != 3:  # 如果通道不在第2维度
            x = x.permute(0, 2, 1, 3)  # [B,H,C,W] → [B,C,H,W]
        # print(x.shape)

        # 输入x的期望形状：[B, C, H, W]
        # 展开为8×8块 → [B, C, H/8, W/8, 8, 8]
        # patches = x.unfold(2, 8, 8).unfold(3, 8, 8)
        #
        # # 计算DCT频谱幅值
        # dct = torch.fft.fft2(patches).abs()
        #
        # # 应用频率掩码 → 仅保留目标频点
        # masked = dct * self.freq_mask
        #
        # # 聚合特征 → [B, C, num_bands]
        # return masked.flatten(2).mean(-1)
        B, C, H, W = x.shape
        patches = x.unfold(2, 8, 8).unfold(3, 8, 8)  # [B, C, H/8, W/8, 8, 8]

        # 计算DCT频谱
        dct = torch.fft.fft2(patches).abs()  # [B, C, H/8, W/8, 8, 8]

        # 应用频率掩码
        masked = dct * self.freq_mask  # 仅保留目标频点

        # 提取每个频带的能量 (关键修正)
        features = []
        for u, v in self.freq_bands:
            band_energy = masked[..., u % 8, v % 8]  # 提取指定频点 → [B, C, H/8, W/8]
            features.append(band_energy.mean(dim=[2, 3]))  # 空间平均 → [B, C]

        # 拼接所有频带特征
        features = torch.stack(features, dim=2)  # [B, C, num_bands]
        return features.view(B, -1)  # [B, C×num_bands]

class WatermarkDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.extractor = DCTBlockExtractor(config['freq_bands'])
        input_dim = config['dct_channels'] * len(config['freq_bands'])

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config['num_bits']),
            nn.Sigmoid()
        )
        self.transform = Compose([
            lambda x: x.float() / 255.0,

            # lambda x: x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        ])

    def forward(self, images):
        x = self.transform(images)
        features = self.extractor(x)
        return self.decoder(features.view(x.size(0), -1))


class AttackAugmentation:
    """鲁棒性增强的数据增强"""

    def __init__(self):
        self.transforms = [
            lambda x: x.filter(ImageFilter.GaussianBlur(1)),
            lambda x: x.resize((x.size[0] // 2, x.size[1] // 2)).resize(x.size),
            lambda x: x.convert('L').convert('RGB')  # 灰度化
        ]

    def __call__(self, img):
        for t in self.transforms:
            if np.random.rand() < 0.3:
                img = t(img)
        return img
