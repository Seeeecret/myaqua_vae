import os
import torch
from diffusers import StableDiffusionPipeline
import argparse


# 加载和融合 LoRA 权重
def load_lora_weights(pipe: StableDiffusionPipeline, lora_path: str, SEC_LORA: str = None, KNOTS=False):
    """
    使用 diffusers 提供的相关接口加载和融合 LoRA。
    根据 .safetensors 文件名调用 pipe.load_lora_weights() 和 pipe.fuse_lora()。
    """
    if KNOTS:
        if lora_path.endswith('.safetensors'):
            from safetensors import safe_open
            lora_weights = {}
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    lora_weights[key] = f.get_tensor(key)
        else:  # 支持.pt或.bin格式
            lora_weights = torch.load(lora_path, map_location="cpu")

        # 获取UNet的原始状态字典
        unet_state_dict = pipe.unet.state_dict()
        updated = 0
        skipped = 0

        # 遍历所有LoRA权重键
        for lora_key in lora_weights:
            # 调整键名格式（移除_0后缀）
            original_key = lora_key.replace("_0.weight", ".weight")

            if original_key in unet_state_dict:
                # 获取原权重和LoRA权重
                original_tensor = unet_state_dict[original_key]
                lora_tensor = lora_weights[lora_key].to(
                    device=original_tensor.device,
                    dtype=original_tensor.dtype
                )

                # 合并权重（假设LoRA是增量权重）
                unet_state_dict[original_key] = original_tensor + lora_tensor
                updated += 1
            else:
                print(f"[WARN] Key mismatch: {original_key}")
                skipped += 1

        # 加载更新后的状态字典
        pipe.unet.load_state_dict(unet_state_dict, strict=False)
        print(f"[INFO] Successfully updated {updated} weights, skipped {skipped} keys")

    elif SEC_LORA is not None:
        pipe.load_lora_weights(SEC_LORA, adapter_name="loranew")
        pipe.load_lora_weights(lora_path, adapter_name="aqualora")
        pipe.set_adapters(["loranew", "aqualora"])
        pipe.fuse_lora(adapter_names=["loranew", "aqualora"], lora_scale=1.0)

    else:
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
    parser.add_argument("--SEC_LORA", type=str, default=None)
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
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    pipe.to("cuda")

    # 如果提供了 LoRA 权重路径，则加载和融合 LoRA 权重
    if args.lora_weights_path:
        load_lora_weights(pipe, args.lora_weights_path, args.SEC_LORA)
        print("LoRA weights loaded and fused.")

    # 生成并保存图像
    generate_images(pipe, args.prompts, args.output_path)


if __name__ == "__main__":
    main()
