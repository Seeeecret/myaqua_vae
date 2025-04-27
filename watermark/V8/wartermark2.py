# 核心代码逻辑（基于watermark.py改造）
from contextlib import contextmanager
from typing import Tuple

from functorch.dim import Tensor
from torchvision import transforms as tvt

import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline
from scipy.stats import norm, truncnorm
from functools import reduce
import numpy as np
from PIL import Image
import torch.nn.functional as F
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
# from diffusers import StableDiffusionInversePipeline  # 新版专用管道


class ImageShield:
    def __init__(self, ch_factor=1, hw_factor=4, height=64, width=64, device="cuda"):
        self.ch = ch_factor  # 通道重复因子（Stable Diffusion潜在空间通道数4）
        self.hw = hw_factor  # 空间重复因子（控制水印密度）
        self.height = height  # 潜在空间高度（原图高/8）
        self.width = width  # 潜在空间宽度（原图宽/8）
        self.device = device

        # 计算水印参数
        self.latentlength = 4 * self.height * self.width
        self.marklength = self.latentlength // (self.ch * self.hw ** 2)
        # self.threshold = self.ch * self.hw ** 2 // 2  # 多数投票阈值

        # 初始化密钥
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)

        # 新增参数校验
        assert 4 % ch_factor == 0, "ch_factor必须能整除4（潜在空间通道数）"
        assert height % hw_factor == 0, "height必须能被hw_factor整除"
        assert width % hw_factor == 0, "width必须能被hw_factor整除"
        self.threshold = self.ch * (self.hw ** 2) // 2
        print(f"初始化的多数投票阈值：{self.threshold}")  # 调试用

    def create_watermark(self):
        # 生成模板比特（TP）
        # TODO：这里可以试着优化成自定义的?
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, self.height // self.hw,
                                              self.width // self.hw]).to(self.device)

        # 扩展模板比特到潜在空间尺寸
        expanded_tp = self.watermark.repeat(1, self.ch, self.hw, self.hw)

        self.repeat_watermark = expanded_tp

        # ChaCha20加密生成水印比特m
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        binary_array = expanded_tp.flatten().cpu().numpy().astype(np.uint8)
        packed_bits = np.packbits(binary_array, axis=0)  # 压缩8倍
        m_byte = cipher.encrypt(packed_bits.tobytes())
        # m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        packed_bits = np.frombuffer(m_byte, dtype=np.uint8)
        m_bit = np.unpackbits(packed_bits).astype(np.uint8)  # 解包恢复原始比特流
        self.m = torch.from_numpy(m_bit).reshape(1, 4, self.height, self.width).float()

        # 生成带水印的初始噪声（截断采样）
        z = self._truncated_sampling(m_bit)
        return z.to(self.device)

    def _truncated_sampling(self, m_bit):
        ppf = norm.ppf([0.0, 0.5, 1.0])
        z = np.zeros(self.latentlength)

        # 防御性校验
        unique_bits = np.unique(m_bit)
        invalid_bits = set(unique_bits) - {0, 1}
        if invalid_bits:
            raise ValueError(f"非法bit值存在: {invalid_bits}")

        # 向量化加速
        bits = m_bit.astype(int)
        lower_bounds = np.where(bits == 0, ppf[0], ppf[1])
        upper_bounds = np.where(bits == 0, ppf[1], ppf[2])

        # 批量采样
        z = truncnorm.rvs(lower_bounds, upper_bounds, loc=0, scale=1, size=self.latentlength)

        return torch.from_numpy(z).reshape(1, 4, self.height, self.width).float().cuda()



    def detect_tamper(self, inverted_noise):
        # 空间分层细化检测（改造自HSTR）
        inverted_bits = (inverted_noise > 0).int()
        CMP = (inverted_bits == self.m.to(inverted_bits.device)).float()

        # 分层空间聚合
        M_ini = CMP.mean(dim=1)  # 通道平均
        refined_mask = self.hierarchical_refinement(M_ini, levels=3)
        return refined_mask

    def hierarchical_refinement(self, mask, levels=3):
        # 分层细化（类似HSTR的空间部分）
        masks = []
        for l in range(levels):
            μ = 2 ** l
            # 分割为μ×μ区域并平均
            pooled = F.avg_pool2d(mask, μ, stride=μ)
            # 上采样回原尺寸
            upsampled = F.interpolate(pooled, scale_factor=μ, mode='nearest')
            masks.append(upsampled)
        return torch.stack(masks).mean(dim=0)

    @contextmanager
    def temporary_scheduler(self,pipe, target_scheduler):
        """临时切换调度器的上下文管理器"""
        original_scheduler = pipe.scheduler
        try:
            pipe.scheduler = target_scheduler.from_config(original_scheduler.config)
            yield pipe
        finally:
            pipe.scheduler = original_scheduler  # 确保恢复原调度器

    # 修改invert_image方法，确保使用正确的pipe
    def invert_image(
            self,
            pipe: StableDiffusionPipeline,
            image: torch.Tensor,
            num_inversion_steps: int = 50,
            guidance_scale: float = 1.0
    ) -> torch.Tensor:

        """使用官方DDIMInverseScheduler实现高质量反演"""
        """增强兼容性的反演函数"""
        # 预处理：确保输入为张量且归一化到[0,1]
        if isinstance(image, Image.Image):
            # 转换为Tensor [0,1]
            image = tvt.ToTensor()(image).unsqueeze(0)  # (1,3,H,W)

            # 调整尺寸匹配模型输入
            if image.shape[2] != pipe.unet.config.sample_size * 8:
                image = tvt.Resize((pipe.unet.config.sample_size * 8, pipe.unet.config.sample_size * 8))(image)
            image.to(device=pipe.device, dtype=pipe.dtype)

        inverted_latents=None

        # 确保pipe使用正确的调度器
        if not isinstance(pipe.scheduler, DDIMInverseScheduler):
            with self.temporary_scheduler(pipe, DDIMInverseScheduler):
                pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

                # 将图像编码到潜在空间
                latents = self._encode_image(pipe, image)

                latents = latents.to(device=pipe.device, dtype=pipe.dtype)

                # 执行DDIM反演
                inverted_latents = pipe(
                    prompt="",
                    negative_prompt="",
                    guidance_scale=guidance_scale,
                    latents=latents,
                    output_type="latent",
                    num_inference_steps=num_inversion_steps,
                    return_dict=False
                )[0]

        # 断言inverted_latents存在
        assert inverted_latents is not None

        return inverted_latents

    def _majority_vote_aggregation_old(self, decoded_tensor: torch.Tensor) -> torch.Tensor:
        """
        多数投票聚合（基于论文的扩散逆过程）
        :param decoded_tensor: 解密后的模板位张量 [1, 4, H, W]
        :return: 聚合后的水印矩阵 [1, C, H//hw, W//hw]
        """
        """与论文代码完全一致的多数投票聚合"""
        # 参数计算，默认 self.ch=1, self.hw=8, self.height=64
        ch_stride = 4 // self.ch
        hw_stride = self.height // self.hw
        ch_list = [ch_stride] * self.ch  # 通道分割策略
        hw_list = [hw_stride] * self.hw  # 空间分割策略

        # 分块维度分解（完全对齐论文split逻辑）
        # 步骤1: 按通道分块 [1,4,H,W] -> [C块数, 块通道数, H, W]
        split_dim1 = torch.cat(
            torch.split(decoded_tensor, ch_list, dim=1),
            dim=0
        )

        # 步骤2: 按高度分块 [C块数, 块通道数, H, W] -> [C*H块数, 块通道数, 块高度, W]
        split_dim2 = torch.cat(
            torch.split(split_dim1, hw_list, dim=2),
            dim=0
        )

        # 步骤3: 按宽度分块 [C*H块数, 块通道数, 块高度, W] -> [总块数, 块通道数, 块高度, 块宽度]
        split_dim3 = torch.cat(
            torch.split(split_dim2, hw_list, dim=3),
            dim=0
        )

        # 多数投票
        # sum over all duplicated blocks [总块数, 块通道数, 块高度, 块宽度] -> [块通道数, 块高度, 块宽度]
        vote = torch.sum(split_dim3, dim=0)
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1

        # 重组为原始水印形状 [1, 4//ch, H//hw, W//hw]
        return vote.unsqueeze(0)


    def _encode_image(self, pipe: StableDiffusionPipeline, image: torch.Tensor) -> torch.Tensor:
        """增强鲁棒性的潜在空间编码方法"""
        # 确保输入在正确设备上
        image = image.to(device=pipe.device)

        # 检查并调整输入范围
        if image.min() >= 0 and image.max() <= 1.0:
            image = 2.0 * image - 1.0  # [0,1] -> [-1,1]
        else:
            raise ValueError("输入图像必须为[0,1]范围张量")

        # 确保数据类型匹配(转换为模型使用的精度)
        image = image.to(dtype=pipe.dtype)  # 通常为torch.float16

        # 检查输入维度 (应为BCHW)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

        # 混合精度编码
        with torch.autocast(device_type='cuda' if 'cuda' in str(pipe.device) else 'cpu'):
            posterior = pipe.vae.encode(image).latent_dist
            # 0.18215 这个常数是 VAE（变分自编码器）的标准差归一化系数
            latents = posterior.mean * 0.18215

        return latents

    def extract_watermark(
            self,
            inverted_latents: torch.Tensor,
            original_shape: Tuple[int, int, int, int] = (1, 4, 64, 64)
    ) -> Tuple[np.ndarray, float]:
        """增强型水印提取与解密"""
        # 噪声二值化
        inverted_bits = (inverted_latents > 0).int()

        # 转换为字节流（确保长度正确）
        bit_stream = inverted_bits.flatten().cpu().numpy().astype(np.uint8)

        packed_bits = np.packbits(bit_stream)

        # ChaCha20解密
        decrypted = self._chacha_decrypt(packed_bits.tobytes())

        # 重组水印矩阵
        decrypted_bits = np.unpackbits(np.frombuffer(decrypted, dtype=np.uint8))

        # 直接重组（需确保self.repeat_watermark.shape形状与decrypted_bits长度匹配）
        decoded = decrypted_bits.reshape(self.repeat_watermark.shape)

        decoded_tensor = torch.from_numpy(decoded)
        # TODO: 待实现多数投票聚合，返回还原水印reversed_watermark的变量 直接计算模板位TP的比特准确率，缺少论文代码中的多数投票聚合 2025.4.8 已实现
        # accuracy = ((decoded == self.repeat_watermark.cpu()).float()).mean()
        # 多数投票聚合
        aggregated_watermark = self._majority_vote_aggregation_old(decoded_tensor)
        # 计算聚合后的准确率（与原始水印比较）
        original_watermark = self.watermark.squeeze(0)  # [1,C,h,w] -> [C,h,w]
        accuracy = (aggregated_watermark == original_watermark.cpu()).float().mean().item()

        return decoded, accuracy

    def _chacha_decrypt(self, data: bytes) -> bytes:
        """优化的ChaCha20解密实现"""
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        try:
            return cipher.decrypt(data)
        except ValueError as e:
            print("ValueError occur in decrypt")
            if len(data) % 64 != 0:
                # 填充处理
                padded_data = data + b'\0' * (64 - len(data) % 64)
                return cipher.decrypt(padded_data)[:len(data)]
            raise e

    # def _majority_vote_aggregation(self, decoded_tensor: torch.Tensor) -> torch.Tensor:
    #     """修正后的多数投票聚合函数"""
    #     # 获取参数
    #     ch_factor = self.ch  # 通道复制因子
    #     hw_factor = self.hw  # 空间复制因子
    #     bs, c, h, w = decoded_tensor.shape  # 输入形状 [1,4,64,64]
    #
    #     # 确保输入维度合法性
    #     assert c == 4, f"潜在空间通道数必须为4，实际得到{c}"
    #     assert h % hw_factor == 0 and w % hw_factor == 0, "HW维度必须能被复制因子整除"
    #
    #     # 分块逻辑 ------------------------------------------------------
    #     # 步骤1：按通道分块 [1,4,64,64] -> [ch_factor, (4/ch_factor), 64, 64]
    #     # 例如 ch_factor=1 时分为 [1,4,64,64]
    #     channel_blocks = torch.chunk(decoded_tensor, ch_factor, dim=1)
    #
    #     # 步骤2：每个通道块再按高度分块 [N, c_sub, h, w] -> [N*hw_factor, c_sub, h_sub, w]
    #     height_blocks = []
    #     for c_block in channel_blocks:
    #         h_blocks = torch.chunk(c_block, hw_factor, dim=2)
    #         height_blocks.extend(h_blocks)
    #
    #     # 步骤3：每个高度块再按宽度分块 [N, c_sub, h_sub, w] -> [N*hw_factor, c_sub, h_sub, w_sub]
    #     spatial_blocks = []
    #     for h_block in height_blocks:
    #         w_blocks = torch.chunk(h_block, hw_factor, dim=3)
    #         spatial_blocks.extend(w_blocks)
    #
    #     # 多数投票聚合 --------------------------------------------------
    #     aggregated = []
    #     for block in spatial_blocks:
    #         # 计算每个块的多数值 [bs, c_sub, h_sub, w_sub]
    #         sum_val = block.sum()
    #         vote = 1 if sum_val > self.threshold else 0
    #         aggregated.append(vote)
    #
    #     # 重组为原始水印形状 [1, (4/ch_factor), (h/hw_factor), (w/hw_factor)]
    #     watermark_shape = [
    #         1,
    #         4 // ch_factor,
    #         h // hw_factor,
    #         w // hw_factor
    #     ]
    #     return torch.tensor(aggregated, device=self.device).reshape(watermark_shape).float()