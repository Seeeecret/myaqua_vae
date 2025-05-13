import math
from contextlib import contextmanager
from typing import Tuple

import torch
import torch.nn.functional as F
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
import numpy as np
from scipy.stats import norm, truncnorm
from torchvision import transforms as tvt
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline

from wartermark import ImageShield


class PolarImageShield(ImageShield):
    def __init__(self,
                 num_rings=8,  # 新增：环形区域数量
                 angular_sectors=32,  # 新增：角度扇区数
                 ch_factor=1,
                 hw_factor=4,
                 height=64,
                 width=64,
                 device="cuda"):
        super().__init__(ch_factor, hw_factor, height, width, device)

        # 极坐标参数
        self.num_rings = num_rings
        self.angular_sectors = angular_sectors
        self.max_radius = min(height, width) // 2

        # 创建极坐标网格
        self.polar_grid = self._create_polar_grid()

    def _create_polar_grid(self):
        """创建极坐标索引映射表"""
        # 笛卡尔坐标系网格
        y, x = torch.meshgrid(
            torch.linspace(-self.max_radius, self.max_radius, self.height),
            torch.linspace(-self.max_radius, self.max_radius, self.width))

        # 转换为极坐标
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x) % (2 * math.pi)

        # 离散化极坐标
        ring_idx = torch.floor(r / (self.max_radius / self.num_rings)).long()
        sector_idx = torch.floor(theta / (2 * math.pi / self.angular_sectors)).long()

        # 限制索引范围
        ring_idx = torch.clamp(ring_idx, 0, self.num_rings - 1)
        sector_idx = torch.clamp(sector_idx, 0, self.angular_sectors - 1)

        return ring_idx, sector_idx

    def create_watermark(self):
        """改进的极坐标水印生成"""
        # 生成环形模板（num_rings x angular_sectors）
        base_watermark = torch.randint(0, 2, [1, self.num_rings, self.angular_sectors]).to(self.device)

        # 扩展模板到原始潜在空间尺寸
        ring_idx, sector_idx = self.polar_grid
        expanded_tp = base_watermark[:, ring_idx, sector_idx]

        # 通道扩展
        expanded_tp = expanded_tp.repeat(1, 4, 1, 1)  # 扩展到4通道

        # 加密流程保持不变
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        binary_array = expanded_tp.flatten().cpu().numpy().astype(np.uint8)
        packed_bits = np.packbits(binary_array, axis=0)
        m_byte = cipher.encrypt(packed_bits.tobytes())
        packed_bits = np.frombuffer(m_byte, dtype=np.uint8)
        m_bit = np.unpackbits(packed_bits).astype(np.uint8)
        self.m = torch.from_numpy(m_bit).reshape(1, 4, self.height, self.width).float()

        # 生成带水印的初始噪声
        z = self._truncated_sampling(m_bit)
        return z.to(self.device)

    def _angle_search(self, inverted_bits):
        """角度搜索对齐"""
        best_angle = 0
        max_correlation = -1
        base_template = self.watermark[0, 0].cpu().numpy()

        # 在0-360度范围内搜索（步长可调）
        for angle in range(0, 360, 5):
            rotated = self._rotate_polar(inverted_bits, angle)
            correlation = np.corrcoef(base_template.flatten(), rotated.flatten())[0, 1]
            if correlation > max_correlation:
                max_correlation = correlation
                best_angle = angle
        return best_angle

    def _rotate_polar(self, tensor, degrees):
        """极坐标下的旋转（角度平移）"""
        # 转换为极坐标表示
        ring_idx, sector_idx = self.polar_grid
        polar_repr = tensor[0, 0][ring_idx, sector_idx]

        # 角度平移（环形独立处理）
        rotate_steps = int(degrees / (360 / self.angular_sectors))
        rotated = torch.roll(polar_repr, shifts=rotate_steps, dims=1)

        # 映射回笛卡尔坐标
        rotated_cart = torch.zeros_like(tensor)
        rotated_cart[0, 0][ring_idx, sector_idx] = rotated
        return rotated_cart

    def extract_watermark(self, inverted_latents, original_shape):
        """改进的提取流程"""
        # 二值化处理
        inverted_bits = (inverted_latents > 0).int().cpu()

        # 角度搜索对齐
        optimal_angle = self._angle_search(inverted_bits)
        aligned_bits = self._rotate_polar(inverted_bits, -optimal_angle)

        # 转换为字节流
        bit_stream = aligned_bits.flatten().numpy().astype(np.uint8)
        packed_bits = np.packbits(bit_stream)

        # 解密流程
        decrypted = self._chacha_decrypt(packed_bits.tobytes())
        decrypted_bits = np.unpackbits(np.frombuffer(decrypted, dtype=np.uint8))

        # 环形聚合
        ring_idx, sector_idx = self.polar_grid
        polar_repr = decrypted_bits.reshape(1, 4, self.height, self.width)[0, 0]
        aggregated = torch.zeros((self.num_rings, self.angular_sectors))

        # 环形多数投票
        for r in range(self.num_rings):
            for s in range(self.angular_sectors):
                mask = (ring_idx == r) & (sector_idx == s)
                aggregated[r, s] = (polar_repr[mask].mean() > 0.5).int()

        # 计算准确率
        original_polar = self.watermark[0, 0][ring_idx, sector_idx].cpu()
        accuracy = (aggregated[ring_idx, sector_idx] == original_polar).float().mean()

        return aggregated.numpy(), accuracy.item()

    # 保留其他原有方法...
