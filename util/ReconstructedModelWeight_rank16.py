import os

import torch
import random
from safetensors.torch import load_file
from safetensors.torch import save_file

# original_lora_param_info = {}
# safetensors_data_path = "./checkpoints/vae_test/origin_ppft_trained/pytorch_lora_weights.safetensors"
# # original_lora_data_lengths = []
# model = load_file(safetensors_data_path)
# for key, value in model.items():
#     param_info = {
#         'shape': value.shape,
#         'length': value.numel()
#     }
#     original_lora_param_info[key] = param_info

first_path = '../generated_samples/sample_1.pth'
kld_weight_05_path1 = '../generated_samples/kld_weight_05/20241101/sample_1.pth'
kld_weight_05_path2 = '../generated_samples/kld_weight_05/20241101/sample_2.pth'
kld_weight_05_path3 = '../generated_samples/kld_weight_05/20241101/sample_3.pth'

kld_weight_05_path4 = '../generated_samples/kld_weight_05/20241105/sample_1.pth'
kld_weight_05_path5 = '../generated_samples/kld_weight_05/20241105/sample_2.pth'
kld_weight_05_path6 = '../generated_samples/kld_weight_05/20241105/sample_3.pth'

kld_weight_0005_path1 = '../generated_samples/kld_weight_0005/20241102/sample_1.pth'
kld_weight_0005_path2 = '../generated_samples/kld_weight_0005/20241102/sample_2.pth'
kld_weight_0005_path3 = '../generated_samples/kld_weight_0005/20241102/sample_3.pth'

kld_weight_00025_path1 = '../generated_samples/kld_weight_00025/20241103/sample_1.pth'
kld_weight_00025_path2 = '../generated_samples/kld_weight_00025/20241103/sample_2.pth'
kld_weight_00025_path3 = '../generated_samples/kld_weight_00025/20241103/sample_3.pth'

kld_weight_0002_path1 = '../generated_samples/kld_weight_0002/20241103/sample_1.pth'
kld_weight_0002_path2 = '../generated_samples/kld_weight_0002/20241103/sample_2.pth'
kld_weight_0002_path3 = '../generated_samples/kld_weight_0002/20241103/sample_3.pth'

pycharm_kld_weight_0002_path1 = '../generated_samples/kld_weight_0002/20241104/sample_1.pth'
pycharm_kld_weight_0002_path2 = '../generated_samples/kld_weight_0002/20241104/sample_2.pth'
pycharm_kld_weight_0002_path3 = '../generated_samples/kld_weight_0002/20241104/sample_3.pth'

sum_kld_weight_0005_path1 = '../generated_samples/sum_kld_weight_0005/20241104/sample_1.pth'
sum_kld_weight_0005_path2 = '../generated_samples/sum_kld_weight_0005/20241104/sample_2.pth'
sum_kld_weight_0005_path3 = '../generated_samples/sum_kld_weight_0005/20241104/sample_3.pth'

rank4_kld_weight_00005_path1 = '../generated_samples/rank4_kld_weight_00005/20241126/sample_1.pth'
rank4_kld_weight_00005_path2 = '../generated_samples/rank4_kld_weight_00005/20241126/sample_2.pth'
rank4_kld_weight_00005_path3 = '../generated_samples/rank4_kld_weight_00005/20241126/sample_3.pth'

rank64_kld_weight_00005_path1 = '../generated_samples/rank64_kld_weight_00005/20241126/sample_1.pth'
rank64_kld_weight_00005_path2 = '../generated_samples/rank64_kld_weight_00005/20241126/sample_2.pth'
rank64_kld_weight_00005_path3 = '../generated_samples/rank64_kld_weight_00005/20241126/sample_3.pth'

rank64_kld_weight_0005_path1 = '../generated_samples/rank64_kld_weight_0005/20241127/sample_1.pth'
rank64_kld_weight_0005_path2 = '../generated_samples/rank64_kld_weight_0005/20241127/sample_2.pth'

rank64_kld_weight_0005_path1_1202 = '../generated_samples/rank64_kld_weight_0005/20241202/sample_1.pth'
rank64_kld_weight_0005_path2_1202 = '../generated_samples/rank64_kld_weight_0005/20241202/sample_2.pth'

rank64_kld_weight_0005_path1_1204 = '../generated_samples/rank64_kld_weight_0005/20241204/sample_1.pth'
rank64_kld_weight_0005_path2_1204 = '../generated_samples/rank64_kld_weight_0005/20241204/sample_2.pth'
rank64_kld_weight_0005_path3_1204 = '../generated_samples/rank64_kld_weight_0005/20241204/sample_2.pth'

SHAO_kld_weight_0005_path1_1208 = '../generated_samples/SHAO_kld_weight_0005/20241208/sample_1.pth'
SHAO_kld_weight_0005_path2_1208 = '../generated_samples/SHAO_kld_weight_0005/20241208/sample_2.pth'
SHAO_kld_weight_0005_path3_1208 = '../generated_samples/SHAO_kld_weight_0005/20241208/sample_3.pth'

rank16_kld_weight_0005_path1_1214 = '../generated_samples/rank16_8bits_kld_weight_0005/20241214/sample_1.pth'
rank16_kld_weight_0005_path2_1214 = '../generated_samples/rank16_8bits_kld_weight_0005/20241214/sample_2.pth'
rank16_kld_weight_0005_path3_1214 = '../generated_samples/rank16_8bits_kld_weight_0005/20241214/sample_3.pth'

encodeDecodeNormalizedData_path = '/mnt/share_disk/dorin/AquaLoRA/output/rank64_alter3_kld_weight_0005_1204/reconstructed_pytorch_lora_weights_37499/reconstructed_data.pth'
SHAO_encodeDecodeNormalizedData_path = '/mnt/share_disk/dorin/AquaLoRA/output/SHAO_alter3_kld_weight_00005_1208/reconstructed_normalized_adapter_model_31/reconstructed_data.pth'

# 选择采样的数据路径
reconstructed_lora_vector = torch.load(rank16_kld_weight_0005_path1_1214)

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

# Set paths
# dataset_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights"
# output_path = os.path.join(dataset_path, "normalized_data")
#
# # List the data files
# data_files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('.pth')]
#
# # Randomly select one data file from the dataset
# data_file = random.choice(data_files)

# Load the data dictionary from the selected file
#  TODO : 设置数据路径
data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank16_8bits_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_18750.pth"
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
save_path = '../output/rank16_8bits_lora_vae_checkpoints_1214'
# save_path = '../output/encodedecode_SHAO_alter3_kld_weight_0005_1208'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# torch.save(restored_state_dict, './output/encodedecode_rank64_alter3_kld_weight_0005_1205/restored_state_dict.pth')
torch.save(restored_state_dict, os.path.join(save_path, 'restored_state_dict.pth'))
# save_file(restored_state_dict, './output/encodedecode_rank64_alter3_kld_weight_0005_1205/pytorch_lora_weights.safetensors')
save_file(restored_state_dict, os.path.join(save_path, 'pytorch_lora_weights.safetensors'))
# Print a confirmation message
print(f"Restored parameters have been saved to 'restored_state_dict.pth'")
print(restored_state_dict)

# print("Sample 2: ")
# print(sample_2_path)