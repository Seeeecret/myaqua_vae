import os
from safetensors.torch import load_file, save_file

def convert_lora_keys(input_dir):
    """
    将指定目录下的 .safetensors 文件中 lora_A.weight 和 lora_B.weight 改为 lora.up.weight 和 lora.down.weight，并另存为新的文件。

    Args:
        input_dir (str): 输入目录，包含 .safetensors 文件。
    """


    for filename in os.listdir(input_dir):
        if filename.endswith("weights.safetensors"):
            input_path = os.path.join(input_dir, filename)

            # 加载 .safetensors 文件
            weights = load_file(input_path)

            # 修改 key 的命名
            updated_weights = {}
            for key, value in weights.items():
                new_key = key
                if "lora_B.weight" in key:
                    new_key = key.replace("lora_B.weight", "lora.up.weight")
                elif "lora_A.weight" in key:
                    new_key = key.replace("lora_A.weight", "lora.down.weight")
                updated_weights[new_key] = value

            # 保存为新文件
            output_path = os.path.join(input_dir, filename.split(".")[0] + "_updown.safetensors")
            save_file(updated_weights, output_path)
            print(f"Processed and saved: {output_path}")

# 使用示例
# input_directory = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/sd-1.5-pokemon-lora-peft"  # 替换为包含 .safetensors 文件的目录路径
# input_directory = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/output/rank8_8bits_lora_vae_checkpoints_0110_2000epoch"  # 替换为包含 .safetensors 文件的目录路径
input_directory = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/sd-1.5-pokemon-lora-peft/new"  # 替换为包含 .safetensors 文件的目录路径

convert_lora_keys(input_directory)
