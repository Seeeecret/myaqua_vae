import torch
import torch.nn as nn
import torch.nn.functional as F

from myVAEdesign3_rank4_0218 import OneDimVAE
from watermark.V2.WatermarkSystem import WatermarkSystem


class OneDimVAE_Watermarked(OneDimVAE):
    def __init__(self, format_data_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.format_data_dict = format_data_dict  # 必须传入格式定义
        # 添加水印系统
        self.watermark = WatermarkSystem(
            latent_dim=self.d_latent,
            watermark_dim=128
        )

        # 水印检测器
        self.w_detector = nn.Sequential(
            nn.Linear(self.d_latent, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

        # 感知损失
        self.perceptual_loss = PerceptualLoss1D()  # 需自定义1D版本

    def inverseNormalization(self, flattened_recons):
        # 移植您提供的函数实现（需支持批量）
        batch_params = []
        for i in range(flattened_recons.size(0)):
            # 单样本处理逻辑...
            """
                   接收VAE重构后的 1D 向量.
                   根据 self.format_data_dict 提供的 mean/std/shape/length 信息,
                   对其进行 "逆归一化 + reshape", 并返回一个字典 restored_state_dict.

                   注意:
                     - 如果你只有单条数据( batch_size=1 ), flattened_recons = [length].
                     - 如果 batch_size>1, 需要额外设计怎么区分/拆分。这里仅演示单条情形。
                   """
            if self.format_data_dict is None:
                print("[warn] self.format_data_dict is None, cannot do inverse normalization.")
                return None

            # 取出所有 key, 并排除 data_dict['data'](若存在)
            data_keys = [k for k in self.format_data_dict.keys() if k != 'data']

            # 如果 flattened_recons 多于1维, 先 squeeze
            # 这里假设只处理 batch_size=1 的情况
            if len(flattened_recons.shape) > 1:
                flattened_recons = flattened_recons.squeeze(0)
                # 现在 flattened_recons 是 [total_length]

            # 收集每个 key 的长度, 用于切分
            lengths = [self.format_data_dict[k]['length'] for k in data_keys]
            total_length = sum(lengths)

            if flattened_recons.shape[0] < total_length:
                print(f"[warn] 你的重建向量长度 {flattened_recons.shape[0]} 小于所需 {total_length}，请检查。")
                print(flattened_recons.shape)
                return None

            # 切分
            split_data = torch.split(flattened_recons, lengths)

            # 开始逐段逆归一化 & reshape
            restored_state_dict = {}
            for i, key in enumerate(data_keys):
                chunk = split_data[i]
                mean = self.format_data_dict[key]['mean']
                std = self.format_data_dict[key]['std']
                shape = self.format_data_dict[key]['shape']

                denormalized_data = chunk * std + mean
                denormalized_data = denormalized_data.reshape(shape)
                restored_state_dict[key] = denormalized_data

            batch_params.append(restored_state_dict)

        return batch_params
    def encode_decode(self, input, apply_wmark=True, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)

        # 添加水印
        if apply_wmark:
            z_w, w = self.watermark(z)
        else:
            z_w = z
            w = torch.zeros((z.size(0), 128), device=z.device)

        recons = self.decode(z_w)
        reccons_z = self.decode(z)
        return recons, input, mu, log_var, z_w, reccons_z, w

    def forward(self, x, **kwargs):
        recons, input, mu, log_var, z_w, reccons_z, w = self.encode_decode(x, **kwargs)

        # 原始重构损失
        padded_x = self.pad_sequence(x).squeeze(1)
        recons_loss = F.mse_loss(reccons_z, padded_x)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # 水印检测损失，在隐空间上检测
        w_pred = self.w_detector(z_w)
        w_loss = F.binary_cross_entropy(w_pred, w)

        # 感知相似性损失
        perc_loss = self.perceptual_loss(
            self.normalize(padded_x),
            self.normalize(recons)
        )

        # 总损失
        total_loss = (
                recons_loss +
                self.kld_weight * kld_loss +
                0.5 * w_loss +
                0.1 * perc_loss
        )

        return {
            "loss": total_loss,
            "recons": recons_loss,
            "kld": kld_loss,
            "wmark": w_loss,
            "perc": perc_loss
        }

    @staticmethod
    def normalize(x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


class PerceptualLoss1D(nn.Module):
    """1D信号感知损失"""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=3, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU()
        )

    def forward(self, x, y):
        feat_x = self.conv_layers(x.unsqueeze(1))
        feat_y = self.conv_layers(y.unsqueeze(1))
        return F.l1_loss(feat_x, feat_y)