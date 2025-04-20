import torch
import torch.nn as nn
import torchvision.transforms as T
from safetensors.torch import load_file, save_file
from scipy.fftpack import dctn, idctn
import numpy as np


class DCTWatermarkEngine:
    def __init__(self,
                 img_size=512,
                 dct_channels=3,
                 freq_bands=[(5, 5), (6, 6), (7, 7)],  # 中高频带坐标
                 seed=2023):
        self.img_size = img_size
        self.dct_channels = dct_channels
        self.freq_bands = freq_bands
        self.rng = np.random.RandomState(seed)

        # 生成DCT基底模板
        self.dct_bases = self._generate_dct_bases()

    def _generate_dct_bases(self):
        """生成预定义的DCT基底模板"""
        bases = []
        for ch in range(self.dct_channels):
            for (u, v) in self.freq_bands:
                base = np.zeros((self.img_size, self.img_size))
                base[u, v] = 1.0
                bases.append((ch, base))
        return bases

    def _msg_to_coeffs(self, msg_bits):
        """将二进制信息映射到DCT系数"""
        assert len(msg_bits) == len(self.dct_bases)
        return [float(bit) * 0.1 - 0.05 for bit in msg_bits]  # 系数范围[-0.05,0.05]

    def embed_to_lora(self, lora_weights, msg_bits, lambda_factor=0.03):
        """
        核心嵌入函数
        :param lora_weights: 原始LoRA权重(safetensors加载的字典)
        :param msg_bits: 二进制水印信息(长度需等于dct_bases数量)
        :param lambda_factor: 水印强度系数
        :return: 含水印的LoRA权重
        """
        # 步骤1：生成水印模板
        coeffs = self._msg_to_coeffs(msg_bits)
        watermark = np.zeros((self.dct_channels, self.img_size, self.img_size))
        for idx, (ch, base) in enumerate(self.dct_bases):
            watermark[ch] += coeffs[idx] * base

        # 步骤2：低秩分解
        tensor_watermark = torch.from_numpy(watermark).float()
        U, S, Vh = torch.linalg.svd(tensor_watermark.reshape(3, -1), full_matrices=False)
        rank = 4  # 选择秩为4的近似
        A = U[:, :rank] @ torch.diag(S[:rank])
        B = Vh[:rank, :]

        # 步骤3：参数融合
        modified_weights = lora_weights.copy()
        for key in modified_weights:
            if 'lora_down' in key:  # 对应LoRA的B矩阵
                delta = lambda_factor * B[:, :modified_weights[key].shape[1]]
                modified_weights[key] += delta.to(modified_weights[key].device)
            elif 'lora_up' in key:  # 对应LoRA的A矩阵
                delta = lambda_factor * A[:, :modified_weights[key].shape[0]]
                modified_weights[key] += delta.T.to(modified_weights[key].device)

        return modified_weights


# 使用示例
if __name__ == "__main__":
    # 初始化引擎
    watermark_engine = DCTWatermarkEngine(
        freq_bands=[(5, 5), (5, 6), (6, 5), (6, 6), (7, 7)],  # 可自定义频率位置
        seed=2023
    )

    # 加载原始LoRA
    orig_lora = load_file("/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/pytorch_lora_weights.safetensors")

    # 生成水印信息（示例使用48位）
    msg_bits = bin(0xDEADBEEF)[2:].zfill(32) + bin(0xCAFE)[2:].zfill(16)  # 48位示例

    # 嵌入水印
    watermarked_lora = watermark_engine.embed_to_lora(orig_lora, msg_bits)

    # 保存带水印的LoRA
    save_file(watermarked_lora, "watermarked_lora.safetensors")
