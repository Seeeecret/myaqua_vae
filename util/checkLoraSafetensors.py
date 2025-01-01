import glob
import os
import scripts
import torch
from torch.utils.tensorboard import SummaryWriter

# import odvae

from safetensors.torch import load_file

filename = "../checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights/pytorch_lora_weights_18750.safetensors"
model = load_file(filename)
no_watermarked_state_dict = model
for key, value in model.items():
    # print("\nkey:", key, "\nvalue:", value)
    print("\nkey:", key)
# 没有合并水印的lora模型的结构
print("\nSummary of no-watermarked LoRA weights:")
no_watermarked_lora_num_layers = {}
no_watermarked_lora_layer_shapes = {}
for key, value in no_watermarked_state_dict.items():
    layer_name = key.split('.')[0]
    if layer_name not in no_watermarked_lora_num_layers:
        no_watermarked_lora_num_layers[layer_name] = 0
        no_watermarked_lora_layer_shapes[layer_name] = []
    no_watermarked_lora_num_layers[layer_name] += 1
    no_watermarked_lora_layer_shapes[layer_name].append(value.shape)

for layer, count in no_watermarked_lora_num_layers.items():
    print(f"Layer: {layer}, Number of parameters: {count}, Shapes: {no_watermarked_lora_layer_shapes[layer]}")