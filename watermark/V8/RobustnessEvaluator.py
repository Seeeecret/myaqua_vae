import argparse
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from tqdm import tqdm
from wartermark2 import ImageShield
from WatermarkEvaluator import set_random_seed, WatermarkEvaluator
import pandas as pd
import os


def save_to_csv(data, output_dir,filename="distortion_args.csv"):
    # 将字典转换为 DataFrame（单行）
    df = pd.DataFrame([data])  # 注意：[data] 使字典变成一行
    # 拼接output_dir和filename
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)




class RobustnessEvaluator(WatermarkEvaluator):
    def __init__(self, args):
        super().__init__(
            output_dir=args.output_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_samples=args.num_samples,
            seed=args.seed,
            dataset=args.dataset,
            model=args.model_path
        )
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.distortion_config = self.parse_distortion_args()

    def parse_distortion_args(self):
        """将命令行参数转换为失真配置字典"""
        return {
            'rotation_degree': self.args.r_degree,
            "rotation_degree_same": self.args.r_degree_same,
            'jpeg_quality': self.args.jpeg_ratio,
            'crop_scale': self.args.crop_scale,
            'crop_ratio': self.args.crop_ratio,
            'gaussian_blur_radius': self.args.gaussian_blur_r,
            'gaussian_noise_std': self.args.gaussian_std,
            'brightness_factor': self.args.brightness_factor
        }

    def apply_distortions(self, clean_img, wm_img, seed):
        """
        应用相同的失真到图像对
        返回: (distorted_clean, distorted_wm)
        """
        # 确保使用相同随机种子
        set_random_seed(seed)

        # 初始化变换序列
        transform_chain = []

        # 1. 旋转
        if self.distortion_config.get('rotation_degree'):
            transform_chain.append(
                transforms.RandomRotation((-self.distortion_config['rotation_degree'],
                                           self.distortion_config['rotation_degree']))
            )

        if self.distortion_config.get('rotation_degree_same'):
            transform_chain.append(
                transforms.RandomRotation((-self.distortion_config['rotation_degree_same'],
                                           self.distortion_config['rotation_degree_same']))
            )

        # 2. 随机裁剪
        if self.distortion_config.get('crop_scale'):
            transform_chain.append(
                transforms.RandomResizedCrop(
                    size=clean_img.size,
                    scale=(self.distortion_config['crop_scale'], 1.0),
                    ratio=(1.0, 1.0)  # 保持原比例
                )
            )

        # 应用可同步的变换
        if transform_chain:
            transform = transforms.Compose(transform_chain)
            clean_img = transform(clean_img)
            wm_img = transform(wm_img)

        # 3. JPEG压缩 (需要单独处理)
        if self.distortion_config.get('jpeg_quality'):
            clean_img = self.jpeg_compress(clean_img, self.distortion_config['jpeg_quality'])
            wm_img = self.jpeg_compress(wm_img, self.distortion_config['jpeg_quality'])

        # 4. 高斯模糊
        if self.distortion_config.get('gaussian_blur_radius'):
            clean_img = clean_img.filter(
                ImageFilter.GaussianBlur(self.distortion_config['gaussian_blur_radius'])
            )
            wm_img = wm_img.filter(
                ImageFilter.GaussianBlur(self.distortion_config['gaussian_blur_radius'])
            )

        # 5. 高斯噪声
        if self.distortion_config.get('gaussian_noise_std'):
            clean_img = self.add_gaussian_noise(clean_img, self.distortion_config['gaussian_noise_std'])
            wm_img = self.add_gaussian_noise(wm_img, self.distortion_config['gaussian_noise_std'])

        # 6. 亮度调整
        if self.distortion_config.get('brightness_factor'):
            brightness_transform = transforms.ColorJitter(
                brightness=self.distortion_config['brightness_factor']
            )
            clean_img = brightness_transform(clean_img)
            wm_img = brightness_transform(wm_img)

        return clean_img, wm_img

    def jpeg_compress(self, img, quality):
        """内存中的JPEG压缩"""
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def add_gaussian_noise(self, img, std):
        """添加高斯噪声"""
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1) * 255
        return Image.fromarray(noisy_img.astype(np.uint8))

    def run_robustness_experiment(self, distortion_types=None):
        """
        执行鲁棒性测试实验
        :param distortion_types: 要测试的失真类型列表，如 ['rotation', 'jpeg']
        """
        # 初始化结果存储
        results = {
            'config': self.distortion_config if distortion_types is None
            else {k: v for k, v in self.distortion_config.items()
                  if k.split('_')[0] in distortion_types},
            'figure': None,
            'auc': 0,
            'tpr_at_1fpr': 0,
            'scores': [],
            "bit_acc": 0,
            'labels': []
        }

        if distortion_types:
            # 只激活指定的失真类型
            active_config = {k: v for k, v in self.distortion_config.items()
                             if k.split('_')[0] in distortion_types}
        else:
            active_config = self.distortion_config

        print(f"\nRunning robustness test with config: {active_config}")

        prompts = self.load_prompts()
        # scores, labels = [], []

        pbar = tqdm(total=2 * self.num_samples, desc="Testing Robustness")

        for i in range(self.num_samples):
            # 生成原始图像对
            clean_img, wm_img = self.generate_image_pair(
                prompts[i],
                seed=self.seed + i
            )

            # 应用失真
            distorted_clean, distorted_wm = self.apply_distortions(
                clean_img, wm_img, seed=self.seed + i
            )

            # 计算失真后的水印的识别bit accuracy
            score_clean = self.compute_bit_acc(distorted_clean)
            score_wm = self.compute_bit_acc(distorted_wm)

            # scores.extend([score_clean, score_wm])
            # labels.extend([0, 1])
            results['scores'].extend([score_clean, score_wm])
            results['labels'].extend([0, 1])

            pbar.update(2)

            # 保存示例图像
            if (i + 1) % 50 == 0:
                self.save_comparison(
                    clean_img, wm_img, distorted_clean, distorted_wm, i + 1
                )

        # 计算指标

        fpr, tpr, thresholds = roc_curve(results['labels'], results['scores'])
        results['auc'] = auc(fpr, tpr)
        results['tpr_at_1fpr'] = tpr[np.argmax(fpr >= 0.01)] if np.any(fpr >= 0.01) else 0

        # 计算指标
        fpr, tpr, thresholds = roc_curve(results['labels'], results['scores'])
        bit_acc = np.mean(results['scores'][1::2])
        results['auc'] = auc(fpr, tpr)
        results['tpr_at_1fpr'] = tpr[np.argmax(fpr >= 0.01)] if np.any(fpr >= 0.01) else 0
        results['bit_acc'] = bit_acc

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=results['auc']).plot()
        plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
        plt.scatter(x=0.01, y=results['tpr_at_1fpr'], c='red', label=f'TPR@1%FPR = {results["tpr_at_1fpr"]:.3f}')

        # 添加标注
        distortion_str = '\n'.join([f'{k}: {v}' for k, v in results['config'].items() if v is not None])
        plt.title(f"ROC Curve (AUC = {results['auc']:.3f})\nDistortions:\n{distortion_str}")
        plt.legend(loc='lower right')

        # 保存图像
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        plot_path = os.path.join(self.output_dir, "plots", f"roc_curve_{self.args.run_name}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        # 打印结果

        print("\nRobustness Test Results")
        print("=" * 40)
        print(f"Configuration: {results['config']}")
        print(f"AUC: {results['auc']:.4f}")
        print(f"TPR@1%FPR: {results['tpr_at_1fpr']:.4f}")
        print(f"Bit accuracy: {bit_acc}")
        print(f"Watermark Score Drop: {np.mean(results['scores'][1::2]) - np.mean(results['scores'][::2]):.4f}")


        return results

    def save_comparison(self, clean, wm, distorted_clean, distorted_wm, idx):
        """保存对比图像"""
        from PIL import ImageDraw
        def add_label(img, text):
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, fill=(255, 0, 0))
            return img

        # 创建对比图
        comparison = Image.new('RGB', (clean.width * 2, clean.height * 2))
        comparison.paste(add_label(clean, "Original Clean"), (0, 0))
        comparison.paste(add_label(wm, "Original Watermarked"), (clean.width, 0))
        comparison.paste(add_label(distorted_clean, "Distorted Clean"), (0, clean.height))
        comparison.paste(add_label(distorted_wm, "Distorted Watermarked"), (clean.width, clean.height))

        os.makedirs(os.path.join(self.output_dir, "comparisons"), exist_ok=True)
        comparison.save(os.path.join(self.output_dir, "comparisons", f"compare_{idx}.png"))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 然后在代码中使用这个device变量而不是硬编码的.cuda()
    parser = argparse.ArgumentParser(description="Watermark Robustness Evaluation")

    # 实验基础参数
    parser.add_argument("--run_name", required=True, help="实验名称标识")
    parser.add_argument("--model_path", default="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset",type=str,default="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/Stable-Diffusion-Prompts")
    parser.add_argument("--output_dir", default="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/eval/result/robust")

    # 失真参数（与Tree-Ring对齐）
    parser.add_argument("--r_degree", type=float, help="旋转角度（如75表示±75度）")
    parser.add_argument("--r_degree_same", type=float, help="旋转角度（如75表示75度）")
    parser.add_argument("--jpeg_ratio", type=int, help="JPEG压缩质量（1-100）")
    parser.add_argument("--crop_scale", type=float, help="裁剪比例（如0.75）")
    parser.add_argument("--crop_ratio", type=float, help="裁剪宽高比（如0.75）")
    parser.add_argument("--gaussian_blur_r", type=int, help="高斯模糊半径")
    parser.add_argument("--gaussian_std", type=float, help="高斯噪声标准差")
    parser.add_argument("--brightness_factor", type=float, help="亮度调整因子")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name

    print(f"实验名: {run_name}\n")

    # 运行实验
    evaluator = RobustnessEvaluator(args)

    # 打印evaluator的distortion_config
    pprint(evaluator.distortion_config,sort_dicts=False)

    result_dict = evaluator.run_robustness_experiment()
    # 调用示例
    save_to_csv(result_dict,output_dir=args.output_dir)