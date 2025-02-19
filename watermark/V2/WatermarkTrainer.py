import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkTrainer:
    def __init__(self, model, sd_integration, device="cuda"):

        self.model = model.to(device)
        self.sd_integration = sd_integration.to(device)
        self.device = device

        # 优化器设置（仅VAE参数）
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=1e-3,
            total_steps=10000
        )

    def train_step(self, x, clean_images, prompts):
        # 将数据移到设备
        x = x.to(self.device)
        clean_images = clean_images.to(self.device)

        # VAE前向传播（获取带水印的recons）
        outputs = self.model(x.to(self.device))
        recons_w = outputs["recons"]  # [B, total_length]

        # 逆归一化获得LoRA参数字典
        lora_params_dict = self.model.inverseNormalization(recons_w)

        # 合并到SD模型并生成图像
        merged_sd = self.sd_integration.merge_lora(lora_params_dict)
        gen_images = merged_sd.generate(prompts)  # 假设返回[B, C, H, W]

        # 计算感知损失（需要实现PerceptualLoss类）
        percep_loss = self.perceptual_loss(gen_images, clean_images)

        # 损失组合（调整权重）
        total_loss = (
                outputs["loss"] +
                0.3 * percep_loss +
                0.1 * outputs["wmark"]
        )

        # 反向传播
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()

        # 返回监控指标
        return {
            "total_loss": total_loss.item(),
            "recons_loss": outputs["recons"].item(),
            "kld_loss": outputs["kld"].item(),
            "percep_loss": percep_loss.item(),
            "wmark_loss": outputs["wmark"].item()
        }

    class PerceptualLoss(nn.Module):
        """实现1D参数的感知损失（需与您任务匹配）"""

        def __init__(self):
            super().__init__()
            self.conv_blocks = nn.Sequential(
                nn.Conv1d(1, 16, 15, stride=4, padding=7),
                nn.ReLU(),
                nn.Conv1d(16, 32, 9, stride=3, padding=4),
                nn.ReLU()
            )

        def forward(self, x, y):
            return F.l1_loss(
                self.conv_blocks(x.unsqueeze(1)),
                self.conv_blocks(y.unsqueeze(1))
            )