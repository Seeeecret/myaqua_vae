import safetensors.torch as st
import torch
import re


# safetensors 文件路径
# file_path = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/stb-diff-1.5-ft-birman-cat/pytorch_lora_weights.safetensors"
file_path = "/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/KnOTS/merged_lora.safetensors"
# file_path = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/pytorch_lora_weights.safetensors"
output_path = "/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/KnOTS/merged_lora_wOUnet.safetensors"

# 加载 safetensors 权重
model_data = st.load_file(file_path)

# 处理 key，去掉前面的 'unet.'，并在 attn1 和 attn2 后添加 'processor'
def normalize_key(key):
    key = key.replace("unet.", "", 1)  # 去掉前缀 "unet."
    # key = re.sub(r"(attn[12])\.", r"\1.processor.", key)  # 在 attn1 和 attn2 后添加 processor
    # 把".lora"换成"_lora"
    # key = key.replace(".lora", "_lora")
    # 把"to_out"换成"to_out.0"
    # key = key.replace("to_out.0", "to_out")
    return key

modified_data = {normalize_key(key): value for key, value in model_data.items()}

# 重新保存文件
st.save_file(modified_data, output_path)

print(f"Modified model saved to {output_path}")