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
            watermark_dim=121
        ).to(self.device)

        # 生成水印测试
        test_z = torch.randn(4, 256).cuda()
        # test_z.to(self.device)
        test_z_w, test_w = self.watermark(test_z)

        print(test_w.shape)  # 应输出 torch.Size([4, 121])

        # 水印检测器
        self.w_detector = nn.Sequential(
            nn.Linear(self.d_latent, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 121),
            nn.Sigmoid()
        )

        # 感知损失
        self.perceptual_loss = PerceptualLoss1D()  # 需自定义1D版本

    def inverseNormalization(self, flattened_recons):
        """
        支持三维输入的批量逆归一化方法
        输入形状: [batch_size, 1, total_length]
        输出: List[Dict[str, Tensor]] 每个元素对应一个样本的参数字典
        """
        if self.format_data_dict is None:
            raise ValueError("必须设置 format_data_dict 以执行逆归一化")

        # 获取所有参数键（排除data键）
        data_keys = [k for k in self.format_data_dict.keys() if k != 'data']

        # 计算各参数块长度
        lengths = [self.format_data_dict[k]['length'] for k in data_keys]
        total_length = sum(lengths)

        # 验证输入维度
        if flattened_recons.dim() == 3:
            # 转换形状为 [batch, features]
            flattened_recons = flattened_recons.squeeze(1)  # [batch, 1695744]


        batch_size = flattened_recons.size(0)

        # 验证长度匹配
        if flattened_recons.size(1) != total_length:
            raise ValueError(
                f"输入长度 {flattened_recons.size(1)} 与格式定义总长度 {total_length} 不匹配"
            )

        # 批量切分（沿特征维度）
        split_data = torch.split(flattened_recons, lengths, dim=1)  # Tuple[[batch, len1], [batch, len2], ...]

        # 预加载元数据到设备
        device = flattened_recons.device
        metas = [
            {
                'key': key,
                'shape': self.format_data_dict[key]['shape'],
                'mean': torch.tensor(self.format_data_dict[key]['mean'], device=device),
                'std': torch.tensor(self.format_data_dict[key]['std'], device=device)
            }
            for key in data_keys
        ]

        # 批量处理
        batch_params = []
        for b in range(batch_size):
            restored = {}
            for i, meta in enumerate(metas):
                # 提取当前样本的参数块
                param_block = split_data[i][b]  # [length_i]

                # 逆归一化
                denormalized = param_block * meta['std'] + meta['mean']

                # 重塑并存储
                restored[meta['key']] = denormalized.reshape(meta['shape'])

            batch_params.append(restored)

        return batch_params
    def encode_decode(self, input, apply_wmark=True, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)

        # 添加水印
        if apply_wmark:
            z_w, w = self.watermark(z)
        else:
            z_w = z
            w = torch.zeros((z.size(0), 121), device=z.device)

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
            "total_loss": total_loss,
            "recons_loss": recons_loss,
            "kld_loss": kld_loss,
            "wmark_loss": w_loss,
            "perc_loss": perc_loss,
            "recons": recons,
            "input": input,
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