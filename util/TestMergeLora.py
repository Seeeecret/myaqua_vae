import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from tqdm import tqdm
import re

def load_lora_weights(pipeline, lora_path, alpha=1.0, device="cuda"):
    """加载自定义格式的LoRA权重并应用到pipeline中"""
    # 加载LoRA权重
    lora_state_dict = load_file(lora_path, device=device)

    # 预处理：收集所有lora层的up和down权重
    lora_pairs = {}
    for key in lora_state_dict.keys():
        if 'lora.down.weight' in key:
            base_key = key.replace('.lora.down.weight', '')
            lora_pairs[base_key] = {'down': key}
        elif 'lora.up.weight' in key:
            base_key = key.replace('.lora.up.weight', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['up'] = key

    # 应用LoRA权重
    for base_key, pair in tqdm(lora_pairs.items(), desc=f"Applying LoRA {lora_path}"):
        if 'down' not in pair or 'up' not in pair:
            print(f"Skipping incomplete pair for {base_key}")
            continue

        # 解析层级结构
        parts = base_key.split('.')
        curr_module = pipeline.unet if parts[0] == 'unet' else pipeline.text_encoder

        try:
            # 遍历模块层级
            for part in parts[1:]:
                # 处理数字索引的情况（如up_blocks.3）
                if part.isdigit():
                    curr_module = curr_module[int(part)]
                else:
                    curr_module = getattr(curr_module, part)

            # 获取原始权重
            original_weight = curr_module.weight
            lora_down = lora_state_dict[pair['down']]
            lora_up = lora_state_dict[pair['up']]

            # 应用LoRA更新
            if len(lora_up.shape) == 2 and len(lora_down.shape) == 2:
                # 标准矩阵乘法情况
                curr_module.weight.data += alpha * torch.mm(lora_up, lora_down)
            else:
                # 处理可能的特殊形状
                print(f"Unexpected shape for {base_key}: down {lora_down.shape}, up {lora_up.shape}")

        except (AttributeError, IndexError, KeyError) as e:
            print(f"Failed to apply LoRA for {base_key}: {str(e)}")
            continue

def merge_loras(base_model_path, lora_path1, lora_path2, output_path, alpha1=1.0, alpha2=1.0):
    """加载基础模型和两个LoRA，合并后保存"""
    # 加载基础模型
    print("Loading base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to("cuda")

    # 加载并应用第一个LoRA
    print("\nApplying first LoRA...")
    load_lora_weights(pipe, lora_path1, alpha=alpha1)

    # 加载并应用第二个LoRA
    print("\nApplying second LoRA...")
    load_lora_weights(pipe, lora_path2, alpha=alpha2)

    # 保存合并后的模型
    print("\nSaving merged model...")
    pipe.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")

# 使用示例
if __name__ == "__main__":
    # 配置路径
    BASE_MODEL_PATH = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5"  # 基础模型路径
    LORA_PATH_1 = "/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/train/Ortho_rank8_bits8_output_0329_local/10101110/pytorch_lora_weights.safetensors"  # 第一个LoRA权重文件
    LORA_PATH_2 = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/juliensimon/stable-diffusion-v1-5-pokemon-lora/w_Unet/pytorch_lora_weights.safetensors"  # 第二个LoRA权重文件
    OUTPUT_PATH = "/baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/test_diffusion"  # 合并后模型保存路径

    # 设置融合权重 (alpha值控制每个LoRA的影响程度)
    ALPHA_1 = 1  # 第一个LoRA的权重
    ALPHA_2 = 1   # 第二个LoRA的权重

    # 执行融合
    merge_loras(
        BASE_MODEL_PATH,
        LORA_PATH_1,
        LORA_PATH_2,
        OUTPUT_PATH,
        alpha1=ALPHA_1,
        alpha2=ALPHA_2
    )