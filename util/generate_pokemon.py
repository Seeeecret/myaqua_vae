import os
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import argparse


# 加载 LoRA 权重
def load_lora_weights(lora_weights_path, model):
    print(f"Loading LoRA weights from {lora_weights_path}")
    lora_weights = load_file(lora_weights_path)

    # 加载 LoRA 权重到 UNet、VAE 和 Text Encoder
    # 对于每个模型部分，如果 LoRA 权重对应该部分，则加载权重
    for layer_name, weight in lora_weights.items():
        # 加载到 UNet
        if layer_name in model.unet.state_dict():
            model.unet.state_dict()[layer_name].data = weight
            print(f"Loaded {layer_name} into UNet with shape {weight.shape}")

        # 加载到 VAE
        elif layer_name in model.vae.state_dict():
            model.vae.state_dict()[layer_name].data = weight
            print(f"Loaded {layer_name} into VAE with shape {weight.shape}")

        # 加载到 Text Encoder
        elif layer_name in model.text_encoder.state_dict():
            model.text_encoder.state_dict()[layer_name].data = weight
            print(f"Loaded {layer_name} into Text Encoder with shape {weight.shape}")

        else:
            print(f"Layer {layer_name} not found in any model component.")
    return model


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

    # 如果提供了 LoRA 权重路径，则加载 LoRA 权重
    if args.lora_weights_path:
        pipe = load_lora_weights(args.lora_weights_path, pipe)
        print("LoRA weights loaded.")

    # 生成并保存图像
    generate_images(pipe, args.prompts, args.output_path)


if __name__ == "__main__":
    main()
