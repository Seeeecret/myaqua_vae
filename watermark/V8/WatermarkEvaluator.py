import os

import numpy as np
import torch
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from wartermark2 import ImageShield  # 替换为实际模块路径

os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"
# 实验设置仿照Tree-ring

class WatermarkEvaluator:
    def __init__(self,output_dir, device="cuda", num_inference_steps=50, num_samples=1000, seed=42,
                 dataset="Gustavosta/Stable-Diffusion-Prompts", model="runwayml/stable-diffusion-v1-5"):
        self.device = device
        self.num_samples = num_samples
        self.seed = seed
        self.dataset = dataset
        self.model = model
        self.output_dir = output_dir
        self.num_inference_steps = num_inference_steps
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 初始化模型和水印系统
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        self.shield = ImageShield(
            ch_factor=1, hw_factor=8,
            height=64, width=64, device=device
        )

    def load_prompts(self):
        """加载Gustavosta/Stable-Diffusion-Prompts数据集"""
        dataset = load_dataset(self.dataset, split="test")
        prompts = [example["Prompt"] for example in dataset]

        # 检查提示词数量是否足够
        required = 2 * self.num_samples
        if len(prompts) < required:
            # 循环填充提示词
            prompts = (prompts * (required // len(prompts) + 1))[:required]

        return prompts[:2 * self.num_samples]

    def generate_image_pair(self, prompt, seed):
        """生成带水印/未带水印的图像对"""
        set_random_seed(seed)

        # 生成未加水印图像
        # clean_latents = self.pipe.get_random_latents()
        clean_latents = torch.randn_like(self.shield.create_watermark(),dtype=self.pipe.dtype).to(self.device)
        clean_image = self.pipe(
            prompt, latents=clean_latents,
            num_inference_steps=self.num_inference_steps,
        ).images[0]

        # 生成带水印图像
        wm_latents = self.shield.create_watermark().to(self.device,dtype=self.pipe.dtype)
        wm_image = self.pipe(
            prompt, latents=wm_latents,
            num_inference_steps=self.num_inference_steps,
        ).images[0]

        return clean_image, wm_image

    def compute_bit_acc(self, image):
        """计算图像与水印模板的相似度"""
        inverted_latents = self.shield.invert_image(self.pipe, image)
        return self.shield.extract_watermark(inverted_latents)[1]

    def run_experiment(self):
        """执行完整实验流程"""
        # 加载提示词
        prompts = self.load_prompts()
        scores, labels = [], []

        os.makedirs(self.output_dir, exist_ok=True)

        pbar = tqdm(total=2 * self.num_samples, desc="Processing Images")

        for i in range(self.num_samples):
            # 生成图像对
            clean_img, wm_img = self.generate_image_pair(
                prompts[i],
                seed=self.seed + i
            )

            # 计算未加水印图像提取出的水印比特的准确率（负样本）
            score_clean = self.compute_bit_acc(clean_img)
            scores.append(score_clean)
            labels.append(0)
            pbar.update(1)

            # 计算加水印图像提取出的水印比特的准确率（正样本）
            score_wm = self.compute_bit_acc(wm_img)
            scores.append(score_wm)
            labels.append(1)
            pbar.update(1)

            # 每隔25张图片保存当前图像对
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(prompts)}\n")
                print(f"clean score: {score_clean}\n")
                print(f"wm score: {score_wm}\n")

                print(f"Image saved at {self.output_dir}/{i + 1}.png\n")
                clean_img.save(os.path.join(self.output_dir, f"clean_{i + 1}.png"))
                wm_img.save(os.path.join(self.output_dir, f"wm_{i + 1}.png"))

        pbar.close()

        # 计算指标
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # 寻找TPR@1%FPR
        target_idx = np.argmax(fpr >= 0.01)
        tpr_at_1fpr = tpr[target_idx] if target_idx > 0 else 0.0

        # 打印结果
        print(f"\nExperiment Results (seed={self.seed})")
        print("=" * 40)
        print(f"AUC: {roc_auc:.4f}")
        print(f"TPR@1%FPR: {tpr_at_1fpr:.4f}")
        print(f"Mean Positive Score: {np.mean(scores[1::2]):.4f}")
        print(f"Mean Negative Score: {np.mean(scores[::2]):.4f}")


def set_random_seed(seed):
    """设置随机种子（兼容Diffusers库）"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if hasattr(torch, "generator"):
        torch.Generator().manual_seed(seed)


if __name__ == "__main__":
    evaluator = WatermarkEvaluator(
        device="cuda",
        num_samples=1000,
        num_inference_steps=50,
        seed=42,
        output_dir="/baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta",
        dataset="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/Stable-Diffusion-Prompts",
        model="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5"
    )
    evaluator.run_experiment()
