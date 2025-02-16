import os

import torch
import random
from safetensors.torch import load_file
from safetensors.torch import save_file

rank8_8bits_kld_weight_0005_8000_path1_1216 = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch/20241216/sample_1.pth'  # 保存生成数据的目录
rank8_8bits_kld_weight_0005_8000_path2_1216 = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch/20241216/sample_2.pth'  # 保存生成数据的目录
rank8_8bits_kld_weight_0005_8000_path3_1216 = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch/20241216/sample_3.pth'  # 保存生成数据的目录

rank8_8bits_partial_kld_weight_0005_150_path1_1219 = '../generated_samples/rank8_8bits_kld_weight_0005_150epoch_partial/20241219/sample_1.pth'  # 保存生成数据的目录
rank8_8bits_partial_kld_weight_0005_150_path2_1219 = '../generated_samples/rank8_8bits_kld_weight_0005_150epoch_partial/20241219/sample_2.pth'  # 保存生成数据的目录
rank8_8bits_partial_kld_weight_0005_150_path3_1219 = '../generated_samples/rank8_8bits_kld_weight_0005_150epoch_partial/20241219/sample_3.pth'  # 保存生成数据的目录

rank8_8bits_partial_kld_weight_0005_8000_path1_1216 = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch_partial/20241219/sample_1.pth'  # 保存生成数据的目录
rank8_8bits_partial_kld_weight_0005_8000_path2_1216 = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch_partial/20241219/sample_2.pth'  # 保存生成数据的目录
rank8_8bits_partial_kld_weight_0005_8000_path3_1216 = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch_partial/20241219/sample_3.pth'  # 保存生成数据的目录


encodeDecodeNormalizedData_path = '/mnt/share_disk/dorin/AquaLoRA/output/rank64_alter3_kld_weight_0005_1204/reconstructed_pytorch_lora_weights_37499/reconstructed_data.pth'
SHAO_encodeDecodeNormalizedData_path = '/mnt/share_disk/dorin/AquaLoRA/output/SHAO_alter3_kld_weight_00005_1208/reconstructed_normalized_adapter_model_31/reconstructed_data.pth'
rank8_encodeDecodeNormalizedData_path = '/mnt/share_disk/dorin/AquaLoRA/output/rank8_8bits_lora_vae_checkpoints_1215/reconstructed_pytorch_lora_weights_18750/reconstructed_data.pth'
rank8_8bits_partial_kld_weight_002_path1_0201 = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/generated_samples/new_rank8_8bits_6000epoch_partial_0201/sample_1.pth'
rank4_4bits_partial_kld_weight_002_path1_0202 = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/generated_samples/new_rank4_4bits_partial_3000epoch_0202/sample_1.pth'
rank4_4bits_partial_better_path1_0204 = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/generated_samples/better_rank4_4bits_partial_3000epoch_0204/sample_1.pth'
rank4_4bits_partial_better_6000_path1_0204 = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/generated_samples/better_rank4_4bits_partial_6000epoch_0204/sample_1.pth'
# TODO: 选择采样的数据路径
reconstructed_lora_vector = torch.load(rank4_4bits_partial_better_6000_path1_0204)

# 打印重建模型参数信息
reconstructed_lora_param_info = {}

if isinstance(reconstructed_lora_vector, dict):
    for key, value in reconstructed_lora_vector.items():
        print(key, value.shape)
        reconstructed_lora_param_info[key] = {'shape': value.shape, 'length': value.numel()}
else:
    print(reconstructed_lora_vector)
    print("Loaded data is not a dictionary. It might be a single Tensor.")
# 打印观察重建的lora权重
# print(reconstructed_lora_vector[0][:1000])
print(reconstructed_lora_vector[:1000])



# Load the data dictionary from the selected file
#  TODO : 设置数据路径
# data_path = "/data/Tsinghua/wuzy/rank4_bits4_dataset/normalized_partial_data/normalized_pytorch_lora_weights_12480.pth"
data_path = "/data/Tsinghua/wuzy/rank4_bits4_output_0203/normalized_partial_data/normalized_pytorch_lora_weights_12480.pth"
# data_path = "/data/Tsinghua/wuzy/rank8_bits8_dataset/normalize_9360/normalized_pytorch_lora_weights_9360.pth"
# data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_18750.pth"
# data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank16_8bits_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_18750.pth"
# data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_37499.pth"

data_dict = torch.load(data_path)

# Remove the 'data' key to get parameter keys
data_keys = [k for k in data_dict.keys() if k != 'data']

# sample_path = encodeDecodeNormalizedData_path
flattened_data = reconstructed_lora_vector

# Get the lengths of each parameter in the order of data_keys
lengths = [data_dict[k]['length'] for k in data_keys]

# Split the flattened_data into chunks according to the parameter lengths

print(f"Input tensor shape: {flattened_data.shape}")
total_length = sum(lengths)
print(f"Total length from split_sizes: {total_length}")
print(f"Flattened data size: {flattened_data.shape[0]}")

# 如果flattened_data不是一个1D张量，则将其第一维去掉
if len(flattened_data.shape) > 1:
    flattened_data = flattened_data.squeeze()
    print(f"Flattened data squeezed to shape: {flattened_data.shape}")
split_data = torch.split(flattened_data, lengths)

# Initialize a dictionary to store the restored parameters
restored_state_dict = {}

# Iterate over each parameter and restore it
for i, key in enumerate(data_keys):
    # Get the corresponding data chunk
    data = split_data[i]
    # Denormalize using the stored mean and std
    mean = data_dict[key]['mean']
    std = data_dict[key]['std']
    denormalized_data = data * std + mean
    # Since the original shape is unknown, keep it as a 1D tensor
    restored_state_dict[key] = denormalized_data
#     重建为原始的形状
    restored_state_dict[key] = denormalized_data.reshape(data_dict[key]['shape'])

# 获取当前系统时间
import time
localtime = time.asctime(time.localtime(time.time()))
# Save the restored state_dict to a file
# TODO: 修改为自己的输出路径
# save_path = '../output/rank8_8bits_lora_vae_checkpoints_1216_8000epoch'
save_path = '../output/better_rank4_4bits_partial_6000epoch_0204'
# save_path = '../output/rank8_8bits_partial_lora_vae_checkpoints_1219_150epoch'

if not os.path.exists(save_path):
    os.makedirs(save_path)


full_weights_path = "/data/Tsinghua/wuzy/rank4_bits4_output_0203/pytorch_lora_weights_12480.safetensors"
full_model_dict = load_file(full_weights_path)  # 返回 {key: tensor} 的字典

# 2. 用 restored_state_dict 中的键值对更新 full_model_dict
for key, value in restored_state_dict.items():
    if key in full_model_dict:
        full_model_dict[key] = value
    else:
        print(f"Warning: key '{key}' not found in the original model, skipping.")


# 3. 保存新的完整权重为 .pth 文件
# merged_model_pth = "merged_model.pth"
torch.save(full_model_dict, os.path.join(save_path, 'restored_state_dict.pth'))
print(f"模型已保存为 restored_state_dict.pth")

# 4. 同时保存为 .safetensors 文件
# merged_model_safetensors = "merged_model.safetensors"
save_file(full_model_dict, os.path.join(save_path, 'pytorch_lora_weights.safetensors'))
print(f"模型已保存为 pytorch_lora_weights.safetensors")

