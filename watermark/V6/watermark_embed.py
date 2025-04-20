import torch
import numpy as np
import json
from scipy.fftpack import dctn, idctn
from safetensors.torch import save_file


class LoRAWatermarker:
    def __init__(self, config_path):
        # 如果传入的是配置文件本身则
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            with open(config_path) as f:
                self.config = json.load(f)

        self.img_size = self.config['img_size']
        self.freq_bands = self.config['freq_bands']
        self.dct_channels = self.config['dct_channels']
        self._validate_config()

    def _validate_config(self):
        assert len(self.freq_bands) * self.dct_channels == self.config['num_bits']

    def _generate_watermark(self, msg_bits):
        watermark = np.zeros((self.dct_channels, self.img_size, self.img_size))
        for idx, (ch, u, v) in enumerate(self._get_mapping()):
            bit = int(msg_bits[idx])
            watermark[ch, u, v] = 0.1 if bit else -0.1
        return torch.from_numpy(watermark).float()

    def _get_mapping(self):
        mapping = []
        for ch in range(self.dct_channels):
            for (u, v) in self.freq_bands:
                mapping.append((ch, u, v))
        return mapping

    def embed(self, lora_weights, msg_bits, output_path, lambda_factor=0.03):
        # 生成水印模板
        watermark = self._generate_watermark(msg_bits)

        # 低秩分解
        U, S, Vh = torch.linalg.svd(watermark.reshape(self.dct_channels, -1))
        rank = self.config['rank']
        A = U[:, :rank] @ torch.diag(S[:rank])
        B = Vh[:rank, :]
        modified_key = 0
        # 注入LoRA参数
        modified = lora_weights.copy()
        for key in modified:
            if 'lora.down' in key:
                target_shape = modified[key].shape

                # 动态调整B矩阵维度
                if len(target_shape) == 2:  # 普通全连接层
                    delta = B[:, :target_shape[1]]
                elif len(target_shape) == 4:  # 卷积层
                    # 将B矩阵重塑为卷积核形状 [out_ch, in_ch, H, W]
                    delta = B[:, :target_shape[1] * target_shape[2] * target_shape[3]]
                    delta = delta.view(target_shape[0], target_shape[1],
                                       target_shape[2], target_shape[3])

                modified[key] += (lambda_factor * delta).to(modified[key].device)

            elif 'lora.up' in key:
                # 类似处理A矩阵
                target_shape = modified[key].shape
                if len(target_shape) == 2:
                    delta = A[:, :target_shape[0]]
                elif len(target_shape) == 4:
                    delta = A[:, :target_shape[0] * target_shape[1] * target_shape[2]]
                    delta = delta.view(target_shape[0], target_shape[1],
                                       target_shape[2], target_shape[3])
                modified[key] += (lambda_factor * delta.T).to(modified[key].device)


            # if 'lora.up' in key:
            #     modified[key] += (lambda_factor * A.T).to(modified[key].device)
            #     modified_key+=1
            # elif 'lora.down' in key:
            #     modified[key] += (lambda_factor * B).to(modified[key].device)
            #     modified_key+=1

        print(f"共修改{modified_key}个参数")
        save_file(modified, output_path)
        return modified


# 配置文件示例 (configs/sd_v1.5_48bit.json)
"""
{
    "img_size": 512,
    "freq_bands": [[5,5], [5,6], [6,5], [6,6], [7,5], [7,7]],
    "dct_channels": 3,
    "num_bits": 48,
    "rank": 4
}
"""
