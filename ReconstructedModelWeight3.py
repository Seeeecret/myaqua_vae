import glob
import os
import scripts
import torch
from torch.utils.tensorboard import SummaryWriter
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

original_lora_param_info = {}
safetensors_data_path = "./checkpoints/georgefen-AquaLoRA-Models/ppft_trained/pytorch_lora_weights.safetensors"
# original_lora_data_lengths = []
model = load_file(safetensors_data_path)
for key, value in model.items():
    param_info = {
        'shape': value.shape,
        'length': value.numel()
    }
    original_lora_param_info[key] = param_info

first_path = '/mnt/share_disk/dorin/AquaLoRA/generated_samples/sample_1.pth'
kld_weight_05_path1 = '/mnt/share_disk/dorin/AquaLoRA/generated_samples/kld_weight_05/20241101/sample_1.pth'
kld_weight_05_path2 = '/mnt/share_disk/dorin/AquaLoRA/generated_samples/kld_weight_05/20241101/sample_2.pth'
kld_weight_05_path3 = '/mnt/share_disk/dorin/AquaLoRA/generated_samples/kld_weight_05/20241101/sample_3.pth'

reconstructed_lora_vector = torch.load(kld_weight_05_path2)



# 打印重建模型参数信息
reconstructed_lora_param_info = {}
# 将 .item() 改为直接使用 .items() 方法进行遍历
# for key, value in reconstructed_lora_vector.items():
#     print(key, value.shape)
#     reconstructed_lora_param_info[key] = {'shape': value.shape, 'length': value.numel()}
if isinstance(reconstructed_lora_vector, dict):
    for key, value in reconstructed_lora_vector.items():
        print(key, value.shape)
        reconstructed_lora_param_info[key] = {'shape': value.shape, 'length': value.numel()}
else:
    print(reconstructed_lora_vector)
    print("Loaded data is not a dictionary. It might be a single Tensor.")

print(reconstructed_lora_vector[0][:1000])
