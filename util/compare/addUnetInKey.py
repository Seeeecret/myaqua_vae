"""
LoRA权重键名批量修改脚本
功能：为safetensors格式的LoRA权重文件所有键添加'unet.'前缀
环境要求：Python 3.8+, 需安装safetensors和torch库
"""

import argparse
from safetensors import safe_open
from safetensors.torch import save_file
import os
import torch


def process_lora_weights(input_path, output_path):
    """
    核心处理函数
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    """
    try:
        # 使用安全模式加载原始权重
        with safe_open(input_path, framework="pt", device="cpu") as f:
            original_tensors = {key: f.get_tensor(key) for key in f.keys()}

            # 创建新键名字典（保留原始元数据）
            new_tensors = {}
            metadata = f.metadata() if hasattr(f, 'metadata') else {}

            for old_key in original_tensors.keys():
                new_key = f"unet.{old_key}"
                new_tensors[new_key] = original_tensors[old_key].clone()

                # 同步元数据（如果存在）
                if metadata and old_key in metadata:
                    metadata[new_key] = metadata.pop(old_key)

                    # 保存处理后的权重文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(new_tensors, output_path, metadata=metadata if metadata else None)
        print(f"Successfully processed and saved to {output_path}")
        print(f"Total keys modified: {len(new_tensors)}")

    except Exception as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA权重键名修改工具')
    # parser.add_argument('-i', '--input', type=str, required=True, help='输入文件路径')
    # parser.add_argument('-o', '--output', type=str, default='modified_model.safetensors',
    #                     help='输出文件路径（默认：modified_model.safetensors ）')

    # args = parser.parse_args()
    input = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/juliensimon/stable-diffusion-v1-5-pokemon-lora/pytorch_lora_weights.safetensors"
    output = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/juliensimon/stable-diffusion-v1-5-pokemon-lora/w_Unet/pytorch_lora_weights.safetensors"

    # input = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/wO_unet/filtered_model.safetensors"
    # output = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/wO_unet/filtered_model_unet.safetensors"

    # 执行处理流程
    process_lora_weights(input, output)