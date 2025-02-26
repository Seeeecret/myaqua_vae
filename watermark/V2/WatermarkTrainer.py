import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkTrainer:
    def __init__(self, model, sd_integration, device="cuda"):

        self.model = model.to(device)
        self.sd_integration = sd_integration.to(device)
        self.device = device
        # self.perceptual_loss = self.PerceptualLoss().to(device)

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

    # def train_step(self, x, clean_images, prompts):
    #     # 将数据移到设备
    #     x = x.to(self.device)
    #     clean_images = clean_images.to(self.device)
    #
    #     # VAE前向传播（获取带水印的recons）
    #     outputs = self.model(x.to(self.device))
    #     recons_w = outputs["recons"]  # [B, total_length]
    #
    #     # 逆归一化获得LoRA参数字典
    #     lora_params_dict = self.model.inverseNormalization(recons_w)
    #
    #     # 合并到SD模型并生成图像
    #     merged_sd = self.sd_integration.merge_lora(lora_params_dict)
    #     gen_images = merged_sd.generate(prompts)  # 假设返回[B, C, H, W]
    #
    #     # 计算感知损失（需要实现PerceptualLoss类）
    #     percep_loss = self.perceptual_loss(gen_images, clean_images)
    #
    #     # 损失组合（调整权重）
    #     total_loss = (
    #             outputs["loss"] +
    #             0.3 * percep_loss +
    #             0.1 * outputs["wmark"]
    #     )
    #
    #     # 反向传播
    #     self.opt.zero_grad()
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #     self.opt.step()
    #     self.scheduler.step()
    #
    #     # 返回监控指标
    #     return {
    #         "total_loss": total_loss.item(),
    #         "recons_loss": outputs["recons"].item(),
    #         "kld_loss": outputs["kld"].item(),
    #         "percep_loss": percep_loss.item(),
    #         "wmark_loss": outputs["wmark"].item()
    #     }

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


class JointTrainer(WatermarkTrainer):
    def __init__(self, model, sd_integration, img_detector, device="cuda"):
        """
        扩展父类初始化：
        - model: 水印VAE模型（必须包含inverseNormalization方法）
        - sd_integration: Stable Diffusion集成模块
        - img_detector: 图像水印检测器
        """
        super().__init__(model, sd_integration, device)

        # 添加图像检测器
        self.img_detector = img_detector.to(device)

        # 重组优化器（同时优化VAE和检测器）
        self.opt = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': 2e-4},
            {'params': self.img_detector.parameters(), 'lr': 5e-5}
        ])

        # 更新学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=1e-3, total_steps=10000
        )

    def train_step(self, batch_data):
        """
        改进的训练步骤（完整闭环）：
        输入结构：
        batch_data = {
            'lora_params': Tensor[B, param_dim],
            'prompts': List[str]
        }
        """
        # 解包数据
        x = batch_data['lora_params'].to(self.device)


        # ================= VAE水印生成 =================
        # 前向传播获取带水印的recons
        outputs = self.model(x)
        recons_w = outputs["recons"]  # [B, total_length]
        # 生成干净图像的伪代码
        origin_input = outputs["input"]
        clean_lora = self.model.inverseNormalization(origin_input)  # 未加水印的原始LoRA参数

        # for i in range(len(clean_lora)):
            # clean_lora[i] = clean_lora[i].to(self.device)
        # with torch.no_grad():

        merged_sd_wo_w = self.sd_integration.merge_lora(clean_lora)
        clean_images = self.sd_integration.generate(batch_data['prompts'])
        print("Clean images shape: ", clean_images.shape)

        # ================= LoRA参数水印检测 =================
        # 直接从解码器输出检测水印（无需生成图像）
        # w_lora_pred = self.model.watermark.detect_from_lora(recons_w, self.model)
        # w_lora_true = self._generate_watermark_template(x.size(0))  # 生成真实水印

        # ================= SD图像生成 =================
        # 逆归一化得到LoRA参数字典
        lora_dict_w = self.model.inverseNormalization(recons_w)

        # 合并LoRA并生成图像（保持梯度）
        # with torch.enable_grad():
        merged_sd_w = self.sd_integration.merge_lora(lora_dict_w)
        gen_images = self.sd_integration.generate(batch_data['prompts'])  # [B, C, H, W]

        # 生成水印标签（与VAE同步）
        with torch.no_grad():
            _, w_img_true = self.model.watermark(
                self.model.reparameterize(*self.model.encode(x))
            )  # [B, watermark_dim]
        # ================= 图像水印检测 =================
        w_img_pred = self.img_detector(gen_images)
        # w_img_true = w_lora_true  # 水印标签应与LoRA一致

        # ================= 损失计算 =================
        # 原始VAE损失
        base_loss = outputs["total_loss"]

        # 水印检测损失（双模态）
        # lora_detect_loss = F.binary_cross_entropy(w_lora_pred, w_lora_true)
        img_detect_loss = F.binary_cross_entropy(w_img_pred, w_img_true)

        # 这两者原来是array，需要将gen_images和clean_images转换为CUDA上的tensor
        if isinstance(gen_images, np.ndarray):
            gen_images = torch.from_numpy(gen_images)
        if isinstance(clean_images, np.ndarray):
            clean_images = torch.from_numpy(clean_images)

        gen_images = gen_images.to(self.device)
        clean_images = clean_images.to(self.device)

        # 生成质量损失
        quality_loss = F.mse_loss(gen_images, clean_images)

        # 总损失组合
        total_loss = (
                base_loss +
                0.5 * (img_detect_loss) +
                0.1 * quality_loss
        )

        # ================= 反向传播 =================
        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.scheduler.step()

        return {
            "total": total_loss.item(),
            "base": outputs["total_loss"].item(),
            "vae_recon_loss": outputs["recons_loss"].item(),
            "vae_kld_loss": outputs["kld_loss"].item(),
            "vae_wmark_loss": outputs["wmark_loss"].item(),
            "vae_percep_loss": outputs["perc_loss"].item(),
            # "detect_lora": lora_detect_loss.item(),
            "img_detect_loss": img_detect_loss.item(),
            "quality": quality_loss.item()
        }

    def _generate_watermark_template(self, batch_size):
        """生成与当前batch匹配的水印标签"""
        w = torch.randint(0, 2, (batch_size, 128), device=self.device).float()
        return self.model.watermark.arnold_transform(w)

    def detect(self, gen_images, lora_params):
        """双模态联合检测接口"""
        # LoRA参数检测
        with torch.no_grad():
            recons = self.model(lora_params)["recons"]
            # w_lora = self.model.watermark.detect_from_lora(recons, self.model)

        # 图像检测
        w_img = self.img_detector(gen_images)

        # 综合决策
        return (w_img > 0.5)