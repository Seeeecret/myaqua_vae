# TestExperiment1.py
import torch
import sys
sys.path.append("../")
sys.path.append("../../")
from torch.utils.data import DataLoader
from pathlib import Path
from watermark.V2.OneDimVAE_Watermarked import OneDimVAE_Watermarked
from watermark.V2.WatermarkSystem import ImageWatermarkDetector3 as ImageWatermarkDetector
from watermark.V2.sd_integration import SDIntegration3
from watermark.V2.WatermarkTrainer import JointTrainer
from TrainScript5_alter3_rank4 import VAEDataset  # 使用您提供的自定义数据集类

# 配置参数
CONFIG = {
    "data_dir": "/data/Tsinghua/wuzy/rank4_bits4_output_0203/normalized_data",
    "format_data_path": "/data/Tsinghua/wuzy/rank4_bits4_output_0203/normalized_data/normalized_pytorch_lora_weights_12480.pth",
    "sd_model_path": "/mnt/nvme0n1/Tsinghua_Node11/Share_Model/stable-diffusion-v1-5",
    "input_length": 1695744,
    "batch_size": 1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def main():
    # ================= 初始化组件 =================
    # 加载格式定义
    format_data_dict = torch.load(CONFIG["format_data_path"])

    # 初始化VAE（注意input_length设置）
    vae = OneDimVAE_Watermarked(
        format_data_dict=format_data_dict,
        input_length=CONFIG["input_length"],  # 关键修改点
        latent_dim=256,
        device=CONFIG["device"]
    ).to(CONFIG["device"])

    # 初始化SD集成模块
    sd_integration = SDIntegration3(
        model_name=CONFIG["sd_model_path"],
        torch_dtype=torch.float32 if CONFIG["device"] == "cpu" else torch.float16,
        device=CONFIG["device"]
    )

    # 初始化图像水印检测器
    img_detector = ImageWatermarkDetector().to(CONFIG["device"])

    # ================= 数据加载 =================
    dataset = VAEDataset(CONFIG["data_dir"])
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: torch.cat([item.unsqueeze(0) for item in batch], dim=0)
    )

    # ================= 训练初始化 =================
    trainer = JointTrainer(
        model=vae,
        sd_integration=sd_integration,
        img_detector=img_detector,
        device=CONFIG["device"]
    )

    # ================= 训练循环 =================
    for batch_idx, batch_data in enumerate(dataloader):
        # 数据预处理
        real_data = batch_data.to(CONFIG["device"])

        # 构造训练批次（假设prompt为固定值）
        train_batch = {
            "lora_params": real_data,
            "prompts": ["a photo of cat"] * CONFIG["batch_size"]
        }

        # 执行训练步骤
        metrics = trainer.train_step(train_batch)

        # 打印训练指标
        print(f"Batch {batch_idx + 1}:")
        print(f"  Total Loss: {metrics['total']:.4f}")
        print(f"  Base Loss: {metrics['base']:.4f}")

        print(f"  VAE Recon Loss: {metrics['vae_recon_loss']:.4f}")
        print(f"  VAE KLD Loss: {metrics['vae_kld_loss']:.4f}")
        print(f"  VAE Watermark Loss: {metrics['vae_wmark_loss']:.4f}")
        print(f"  VAE Perceptual Loss: {metrics['vae_percep_loss']:.4f}")
        print(f"  Image Detect Loss: {metrics['img_detect_loss']:.4f}")

        # 每50个batch验证一次
        if (batch_idx + 1) % 50 == 0:
            # 生成测试样本
            with torch.no_grad():
                test_lora = vae.sample(batch=1)
                lora_dict = vae.inverseNormalization(test_lora)

                # 生成图像并检测
                sd_integration.merge_lora(lora_dict)
                gen_images = sd_integration.generate("a photo of dog")
                validity = trainer.detect(gen_images, test_lora)

            print(f"Validation - Watermark Valid: {validity.item():.2%}")


if __name__ == "__main__":
    main()