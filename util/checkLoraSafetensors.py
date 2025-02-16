# import glob
# import os
# import scripts
# import torch
# from torch.utils.tensorboard import SummaryWriter
# from safetensors.torch import load_file
# from collections import defaultdict, Counter
#
# # 文件路径
# filename = "../checkpoints/sd-1.5-pokemon-lora-peft/pytorch_lora_weights.safetensors"
#
# # 加载模型权重
# model = load_file(filename)
# no_watermarked_state_dict = model
#
# # 初始化计数器
# total_keys = 0
# key_type_counter = Counter()
#
# # 打印所有键并进行计数分类
# print("Listing all keys in the model:")
# for key in model.keys():
#     print(f"\nkey: {key}")
#     total_keys += 1
#
#     # 提取键的类型部分
#     # 假设类型是键名的最后几部分，例如 'lora_A.weight'
#     # 你可以根据实际情况调整分割方式
#     key_parts = key.split('.')
#     if len(key_parts) >= 4:
#         # 例如，对于 'unet.down_blocks.0.attentions.0.proj_in.lora_A.weight'
#         # 类型可以定义为 'proj_in.lora_A.weight'
#         key_type = '.'.join(key_parts[-3:])
#     else:
#         key_type = key  # 如果结构不同，保留整个键名作为类型
#
#     key_type_counter[key_type] += 1
#
# # 打印总键数
# print(f"\nTotal number of keys: {total_keys}\n")
#
# # 打印每种类型的键数量
# print("Key types and their counts:")
# for key_type, count in key_type_counter.items():
#     print(f"Type: {key_type}, Count: {count}")
#
# # lora模型的结构
# print("\nSummary of LoRA weights:")
# no_watermarked_lora_num_layers = {}
# no_watermarked_lora_layer_shapes = {}
#
# for key, value in no_watermarked_state_dict.items():
#     layer_name = key.split('.')[0]
#     if layer_name not in no_watermarked_lora_num_layers:
#         no_watermarked_lora_num_layers[layer_name] = 0
#         no_watermarked_lora_layer_shapes[layer_name] = []
#     no_watermarked_lora_num_layers[layer_name] += 1
#     no_watermarked_lora_layer_shapes[layer_name].append(value.shape)
#
# for layer, count in no_watermarked_lora_num_layers.items():
#     print(f"Layer: {layer}, Number of parameters: {count}, Shapes: {no_watermarked_lora_layer_shapes[layer]}")




import glob
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
from collections import defaultdict, Counter

# 文件路径
# filename = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/sd-1.5-pokemon-lora-peft/new/pytorch_lora_weights.safetensors"
# filename = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/stb-diff-1.5-ft-birman-cat/pytorch_lora_weights.safetensors"
filename = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/stb-diff-1.5-ft-birman-cat/pytorch_lora_weights_wo_unet.safetensors"
# filename = "/data/Tsinghua/wuzy/juliensimon/stable-diffusion-v1-5-pokemon-lora/pytorch_lora_weights.safetensors"
# filename = "/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/stable-diffusion-v1-5-lora/pytorch_lora_weights.safetensors"

# 加载模型权重
model = load_file(filename)
no_watermarked_state_dict = model

# 1. 先列举并统计所有键
total_keys = 0
key_type_counter = Counter()

print("Listing all keys in the model:")
for key in model.keys():
    print(f"\nkey: {key}")
    total_keys += 1

    # 假设类型是键名的最后几部分，例如 'lora_A.weight'
    key_parts = key.split('.')
    if len(key_parts) >= 4:
        # 如 'unet.down_blocks.0.attentions.0.proj_in.lora_A.weight'
        # 将类型定义为 'proj_in.lora_A.weight'
        key_type = '.'.join(key_parts[-3:])
    else:
        # 如果结构不同或分割段数不足，保留整个键名
        key_type = key

    key_type_counter[key_type] += 1

print(f"\nTotal number of keys: {total_keys}\n")

print("Key types and their counts:")
for key_type, count in key_type_counter.items():
    print(f"Type: {key_type}, Count: {count}")

# 2. 更加精细地统计（分组）LoRA权重结构
print("\nSummary of LoRA weights (grouped by block/layer):")

# 用 defaultdict(list) 存储：分组 -> [形状列表]
no_watermarked_lora_groups = defaultdict(list)

for key, value in no_watermarked_state_dict.items():
    # 这一步可以根据实际需要调整:
    # - 如果只去掉最后一段 (通常是.weight或.bias)，就写[:-1]
    # - 如果要再往前聚合，可以写[:-2] 等
    group_name = ".".join(key.split('.')[:-1])  # 去掉最后一段

    no_watermarked_lora_groups[group_name].append(value.shape)

for group, shapes in no_watermarked_lora_groups.items():
    print(f"Group: {group}, Number of parameters: {len(shapes)}, Shapes: {shapes}")


# 3. 如果你想“逐个”打印每个 Key 以及对应 tensor 的形状，可以像这样做：
print("\nDetail of each key's shape:")
for key, value in no_watermarked_state_dict.items():
    print(f"Key: {key}, Shape: {value.shape}")

# 4. 计算打印总参数数量
total_params = sum(tensor.numel() for tensor in no_watermarked_state_dict.values())
print(f"\nTotal number of parameters: {total_params}")