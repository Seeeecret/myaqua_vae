from contextlib import contextmanager
from typing import Tuple

from functorch.dim import Tensor
from torchvision import transforms as tvt
import math
import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline
from scipy.stats import norm, truncnorm
from functools import reduce
import numpy as np
from PIL import Image
import torch.nn.functional as F
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

class PolarTransformer:
    """极坐标与笛卡尔坐标转换工具"""


    def __init__(self, input_shape: Tuple[int, int]):
        self.height, self.width = input_shape
        self.center = (self.width // 2, self.height // 2)
        self.max_radius = math.sqrt(
            (self.width / 2) ** 2 + (self.height / 2) ** 2
        )  # 实际最大半径

    def to_polar(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """优化后的极坐标转换"""
        # 生成笛卡尔网格
        y, x = torch.meshgrid(
            torch.arange(self.height, device=img_tensor.device),
            torch.arange(self.width, device=img_tensor.device),
            indexing='ij'
        )
        x_centered = x.float() - self.center[0]
        y_centered = y.float() - self.center[1]

        # 计算极坐标参数
        r = torch.sqrt(x_centered ** 2 + y_centered ** 2)
        theta = torch.atan2(y_centered, x_centered)

        # 归一化到[-1,1]范围（grid_sample要求）
        r_norm = 2 * (r / self.max_radius) - 1  # [-1,1]覆盖整个图像
        theta_norm = theta / math.pi  # [-1,1]对应[-π, π]

        # 构建采样网格
        grid = torch.stack([theta_norm, r_norm], dim=-1)
        grid = grid.unsqueeze(0).expand(img_tensor.size(0), -1, -1, -1)

        # 使用双三次插值提高精度
        return F.grid_sample(
            img_tensor,
            grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=False
        )

    def to_cartesian(self, polar_tensor: torch.Tensor) -> torch.Tensor:
        """优化后的笛卡尔逆转换"""
        # 生成极坐标网格
        theta = torch.linspace(-math.pi, math.pi, polar_tensor.size(3))
        r = torch.linspace(0, self.max_radius, polar_tensor.size(2))
        theta_grid, r_grid = torch.meshgrid(theta, r, indexing='xy')

        # 转换为笛卡尔坐标
        x = r_grid * torch.cos(theta_grid) + self.center[0]
        y = r_grid * torch.sin(theta_grid) + self.center[1]

        # 归一化到[-1,1]
        x_norm = 2 * (x / (self.width - 1)) - 1
        y_norm = 2 * (y / (self.height - 1)) - 1

        # 构建采样网格
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.unsqueeze(0).to(polar_tensor.device)

        # 逆向采样
        return F.grid_sample(
            polar_tensor,
            grid,
            mode='bicubic',
            padding_mode='border',
            align_corners=False
        )

    def _create_reverse_grid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """生成笛卡尔->极坐标的逆变换网格"""
        # 转换为极坐标参数
        r = torch.sqrt(x ** 2 + y ** 2) * self._max_radius()
        theta = torch.atan2(y, x)

        # 归一化到采样范围
        theta_norm = theta / (2 * np.pi)  # [-0.5, 0.5]
        r_norm = r / self._max_radius()  # [0, 1]

        # 组合为grid_sample需要的格式
        return torch.stack([theta_norm, r_norm], dim=-1).unsqueeze(0)

    def _max_radius(self) -> float:
        """计算最大有效半径（防止边缘溢出）"""
        return np.sqrt(2) / 2  # 单位圆内接正方形

    def _create_grid(self, r, theta, h, w,b):
        """生成极坐标采样网格"""
        # 归一化半径和角度
        r_norm = 2 * r / max(h, w) - 1
        theta_norm = theta / np.pi  # [-1, 1]
        grid = torch.stack([theta_norm, r_norm], dim=-1)
        return grid.unsqueeze(0).expand(b, -1, -1, -1)
