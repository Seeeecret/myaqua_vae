import random

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
def seed_everything(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class WatermarkedImageGenerator:
    def __init__(self, base_model="runwayml/stable-diffusion-v1-5",
                 lora_path="watermarked_lora.safetensors",
                 device="cuda"):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)

        # 加载水印LoRA
        lora_weights = load_file(lora_path)
        self.pipe.load_lora_weights(lora_weights)
        # self.pipe.unet.load_state_dict(lora_weights, strict=False)


    def generate_batch(self, prompts, output_dir="dataset", num_images=1000):
        from pathlib import Path
        Path(output_dir).mkdir(exist_ok=True)

        # 生成带水印图像
        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            image = self.pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            image.save(f"{output_dir}/watermarked_{i:04d}.png")

    def generate_batch_ori(self, prompts, output_dir="dataset", num_images=1000):
        from pathlib import Path
        Path(output_dir).mkdir(exist_ok=True)
        # 生成带水印图像
        for i in range(num_images):
            prompt = prompts[i % len(prompts)]
            image = self.pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            image.save(f"{output_dir}/original_{i:04d}.png")

        # 生成对照图像（无水印）
        # self.pipe.unet.load_state_dict(load_file("original_lora.safetensors"), strict=False)
        # for i in range(num_images):
        #     prompt = prompts[i % len(prompts)]
        #     image = self.pipe(
        #         prompt,
        #         num_inference_steps=50,
        #         guidance_scale=7.5
        #     ).images[0]
        #     image.save(f"{output_dir}/original_{i:04d}.png")


# 使用示例
if __name__ == "__main__":
    seed_everything()

    generator = WatermarkedImageGenerator(base_model='/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5',
                                          lora_path='/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/pytorch_lora_weights.safetensors',
                                          device='cuda:2')
    # 从../../evaluation/prompts.txt加载生成文本
    with open("./prompt.txt") as f:
        prompts = f.readlines()

    generator.generate_batch_ori(prompts,output_dir='/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V6output/ori/sampleDataset', num_images=1000)
