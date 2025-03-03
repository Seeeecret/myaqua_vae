import os
import torch
from diffusers import StableDiffusionPipeline
import argparse


# 加载和融合 LoRA 权重
def load_lora_weights(pipe: StableDiffusionPipeline, lora_path: str):
    """
    使用 diffusers 提供的相关接口加载和融合 LoRA。
    根据 .safetensors 文件名调用 pipe.load_lora_weights() 和 pipe.fuse_lora()。
    """
    print(f"[INFO] Loading LoRA weights from: {lora_path}")
    # 下面这行会尝试从指定路径加载名为 "pytorch_lora_weights.safetensors" 的文件，
    # 如果需要自定义文件名，可在 load_lora_weights() 中手动指定。
    pipe.load_lora_weights(lora_path)
    # 将加载的 LoRA 融合到基础模型中
    pipe.fuse_lora(lora_scale=1.0)


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Pokémon images with or without LoRA weights.")
    parser.add_argument('--lora_weights_path', type=str, help="Path to the LoRA weights file (optional)")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the generated images")
    parser.add_argument('--prompts', type=str, nargs='+', required=True, help="List of prompts for generation")
    return parser.parse_args()


# 生成并保存图像
def generate_images(model, prompts, output_path, num_images=10):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, prompt in enumerate(prompts):
        # 生成图像
        print(f"Generating image {i + 1} for prompt: {prompt}")
        with torch.no_grad():
            output = model(prompt, num_inference_steps=50)

        # 输出返回值的所有键，以帮助调试
        print(f"Output keys: {output.keys()}")

        # 获取图像（检查返回值中是否有 'images' 键）
        if 'images' in output:
            images = output['images']
        else:
            print(f"Error: No 'images' found in the output for prompt: {prompt}")
            continue

        # 保存图像
        image_path = os.path.join(output_path, f"pokemon_{i + 1}_{prompt}.png")
        images[0].save(image_path)
        print(f"Saved image {i + 1} at {image_path}")


# 主函数
def main():
    args = parse_args()

    # 加载 Stable Diffusion v1.5 模型
    print("Loading Stable Diffusion model...")
    model_id = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,safety_checker=None)
    pipe.to("cuda")

    # 如果提供了 LoRA 权重路径，则加载和融合 LoRA 权重
    if args.lora_weights_path:
        load_lora_weights(pipe, args.lora_weights_path)
        print("LoRA weights loaded and fused.")

    # 生成并保存图像
    generate_images(pipe, args.prompts, args.output_path)


if __name__ == "__main__":
    main()
