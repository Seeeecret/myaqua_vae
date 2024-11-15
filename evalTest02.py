import glob
import os
import scripts
import torch
from torch.utils.tensorboard import SummaryWriter
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

original_lora_param_info = {}
safetensors_data_path = "./checkpoints/vae_test/origin_ppft_trained/pytorch_lora_weights.safetensors"
# original_lora_data_lengths = []
model = load_file(safetensors_data_path)
for key, value in model.items():
    param_info = {
        'shape': value.shape,
        'length': value.numel()
    }
    original_lora_param_info[key] = param_info

reconstructed_lora_vector = torch.load('/mnt/share_disk/dorin/AquaLoRA/output/sum_kld_weight_0005/vae_final.pth')
# 打印重建模型参数信息
reconstructed_lora_param_info = {}
# 将 .item() 改为直接使用 .items() 方法进行遍历
# for key, value in reconstructed_lora_vector.items():
#     print(key, value.shape)
#     reconstructed_lora_param_info[key] = {'shape': value.shape, 'length': value.numel()}
if isinstance(reconstructed_lora_vector, dict):
    for key, value in reconstructed_lora_vector.items():
        print(key, value.shape)
        print(value)
        reconstructed_lora_param_info[key] = {'shape': value.shape, 'length': value.numel()}
else:
    print(reconstructed_lora_vector)
    print("Loaded data is not a dictionary. It might be a single Tensor.")

# print(reconstructed_lora_vector[0][:1000])
