import os

import numpy as np
from diffusers import StableDiffusionPipeline
from scipy.stats import binom
from tqdm import tqdm
from datasets import load_dataset
from math import comb

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

from wartermark2 import ImageShield
def calculate_fpr(tau, k):
    sum_combinations = sum(comb(k, i) for i in range(tau + 1, k + 1))
    fpr = 1 / (2 ** k) * sum_combinations
    return fpr


def get_threshold(k, fpr):
    tau = 0
    while calculate_fpr(tau, k) > fpr:
        tau += 1
    return tau

class RealTimeWatermarkEvaluator:
    def __init__(self, pipe, shield, output_dir,num_samples=1000, target_fpr=1e-6,
                 dataset="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/Stable-Diffusion-Prompts"):
        """
        :param pipe: Stable Diffusion管道
        :param shield: 水印模块实例
        :param num_samples: 生成样本数
        :param target_fpr: 目标误报率
        """
        self.pipe = pipe
        self.shield = shield
        self.num_samples = num_samples
        self.target_fpr = target_fpr
        self.dataset = dataset
        self.output_dir = output_dir

        # 计算检测阈值
        self.k = shield.marklength  # 从shield获取水印长度
        self.n_repeats = shield.ch * (shield.hw ** 2)
        # self.tau = get_threshold(self.k, target_fpr) / self.k
        self.p_effective = self._compute_effective_p()
        self.tau = self._compute_threshold()

        print(f"配置：水印长度={self.k}，重复次数={self.n_repeats}，τ={self.tau}")

    def _compute_effective_p(self):
        """计算多数投票后的误码率"""
        n = self.n_repeats
        threshold = n // 2
        # p_effective = sum(binom.pmf(k, n, 0.5) for k in range(0, threshold + 1))
        p_effective = binom.cdf(threshold, n, 0.5)
        print(f"实际有效p值: {1 -p_effective:.2e}")

        return 1 - p_effective

    def load_prompts(self):
        """加载Gustavosta/Stable-Diffusion-Prompts数据集"""
        dataset = load_dataset(self.dataset, split="test")
        prompts = [example["Prompt"] for example in dataset]

        # 检查提示词数量是否足够
        required = 2 * self.num_samples
        if len(prompts) < required:
            # 循环填充提示词
            prompts = (prompts * (required // len(prompts) + 1))[:required]

        return prompts[:self.num_samples]
    def _compute_threshold(self):
        """二分搜索寻找满足FPR的τ"""
        left, right = 0, self.k
        best_tau = 0
        while left <= right:
            mid = (left + right) // 2
            fpr = 1 - binom.cdf(mid, self.k, self.p_effective)
            if fpr <= self.target_fpr:
                best_tau = mid
                left = mid + 1
            else:
                right = mid - 1
        return best_tau

    def generate_and_evaluate(self, prompts):
        """
        实时生成带水印图片并评估
        :param prompts: 提示词列表（长度需>=num_samples）
        """
        tp = 0
        error_dist = []

        for i in tqdm(range(self.num_samples), desc="实时评估"):
            # 1. 生成带水印图像
            wm_latents = self.shield.create_watermark()
            wm_image = self.pipe(
                prompts[i],
                latents=wm_latents,
                num_inference_steps=50,
            ).images[0]

            # 2. 立即提取水印
            inverted = self.shield.invert_image(self.pipe, wm_image)
            _, accuracy = self.shield.extract_watermark(inverted)

            # 3. 计算错误位数
            error_bits = int((1 - accuracy) * self.k)
            error_dist.append(error_bits)

            # 4. 统计TP
            if error_bits <= self.tau:
                print("检测到水印")
                tp += 1
            # if accuracy > self.tau:
            #     print("检测到水印")
            #     tp += 1

            # 可选：每100张保存样本
            if i % 100 == 0:
                save_path=os.path.join(output_dir,f"samples/wm_{i}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                wm_image.save(save_path)

        # 计算最终指标
        tpr = tp / self.num_samples
        print(f"\n评估结果（FPR≤{self.target_fpr:.1e}）")
        print("=" * 40)
        print(f"TPR: {tpr:.4f}")
        print(f"错误位数分布：均值={np.mean(error_dist):.1f} ± {np.std(error_dist):.1f}")

        return tpr, error_dist


# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 初始化SD管道和水印模块
    pipe = StableDiffusionPipeline.from_pretrained("/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5",
                                                   safety_checker=None).to("cuda")
    shield = ImageShield(ch_factor=1, hw_factor=8, height=64, width=64)

    # 准备提示词

    # prompts = load_dataset("Gustavosta/Stable-Diffusion-Prompts")["train"]["Prompt"]
    # dataset = load_dataset("/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/Stable-Diffusion-Prompts", split="test")
    output_dir = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/eval/result/10e-1"
    os.makedirs(output_dir, exist_ok=True)

    # 运行实时评估
    evaluator = RealTimeWatermarkEvaluator(
        pipe=pipe,
        shield=shield,
        num_samples=1000,
        output_dir=output_dir,
        dataset='/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/Stable-Diffusion-Prompts',
        target_fpr=1e-6
    )
    prompts = evaluator.load_prompts()
    tpr, errors = evaluator.generate_and_evaluate(prompts)

    # 可视化结果
    import matplotlib.pyplot as plt

    plt.hist(errors, bins=30)
    plt.axvline(evaluator.tau, color='r', linestyle='--', label=f'阈值τ={evaluator.tau}')
    plt.xlabel('错误比特数')
    plt.ylabel('频次')
    plt.title('实时水印检测错误分布')
    plt.legend()
    os.makedirs(os.path.dirname(os.path.join(output_dir,"/results/")), exist_ok=True)
    plt.savefig(os.path.join(output_dir,"/results/error_dist.png"))
