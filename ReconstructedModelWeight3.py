import os

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

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

first_path = './generated_samples/sample_1.pth'
kld_weight_05_path1 = './generated_samples/kld_weight_05/20241101/sample_1.pth'
kld_weight_05_path2 = './generated_samples/kld_weight_05/20241101/sample_2.pth'
kld_weight_05_path3 = './generated_samples/kld_weight_05/20241101/sample_3.pth'

kld_weight_05_path4 = './generated_samples/kld_weight_05/20241105/sample_1.pth'
kld_weight_05_path5 = './generated_samples/kld_weight_05/20241105/sample_2.pth'
kld_weight_05_path6 = './generated_samples/kld_weight_05/20241105/sample_3.pth'

kld_weight_0005_path1 = './generated_samples/kld_weight_0005/20241102/sample_1.pth'
kld_weight_0005_path2 = './generated_samples/kld_weight_0005/20241102/sample_2.pth'
kld_weight_0005_path3 = './generated_samples/kld_weight_0005/20241102/sample_3.pth'

kld_weight_00025_path1 = './generated_samples/kld_weight_00025/20241103/sample_1.pth'
kld_weight_00025_path2 = './generated_samples/kld_weight_00025/20241103/sample_2.pth'
kld_weight_00025_path3 = './generated_samples/kld_weight_00025/20241103/sample_3.pth'

kld_weight_0002_path1 = './generated_samples/kld_weight_0002/20241103/sample_1.pth'
kld_weight_0002_path2 = './generated_samples/kld_weight_0002/20241103/sample_2.pth'
kld_weight_0002_path3 = './generated_samples/kld_weight_0002/20241103/sample_3.pth'

pycharm_kld_weight_0002_path1 = './generated_samples/kld_weight_0002/20241104/sample_1.pth'
pycharm_kld_weight_0002_path2 = './generated_samples/kld_weight_0002/20241104/sample_2.pth'
pycharm_kld_weight_0002_path3 = './generated_samples/kld_weight_0002/20241104/sample_3.pth'

sum_kld_weight_0005_path1 = './generated_samples/sum_kld_weight_0005/20241104/sample_1.pth'
sum_kld_weight_0005_path2 = './generated_samples/sum_kld_weight_0005/20241104/sample_2.pth'
sum_kld_weight_0005_path3 = './generated_samples/sum_kld_weight_0005/20241104/sample_3.pth'

# 选择采样的数据路径
reconstructed_lora_vector = torch.load(kld_weight_00025_path2)

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
print(reconstructed_lora_vector[0][:1000])

start = 0
reconstructed_lora_weights = {}
for layer_name, param_info in original_lora_param_info.items():
    # 在lora权重中，key为layer_name代表某一层的名字，param_info['length']代表这一层的参数个数, param_info['shape']代表这一层参数的shape
    # 从total_lora_data中取出这一层的参数
    end = start + param_info['length']
    layer_weight_vector = reconstructed_lora_vector[0][start:end]
    layer_weight_matrix = layer_weight_vector.view(param_info['shape'])
    reconstructed_lora_weights[layer_name] = layer_weight_matrix
    start = end

# 打印观察重建的lora权重
for layer_name, layer_weight in reconstructed_lora_weights.items():
    print(f"Layer: {layer_name}, Shape: {layer_weight.shape}")
    print(layer_weight)



new_path = "./checkpoints/vae_test/my_ppft_trained_kld_weight_00025"
os.makedirs(new_path, exist_ok=True)
# 保存重建的lora权重为safetensors
save_file(reconstructed_lora_weights, os.path.join(new_path,'pytorch_lora_weights.safetensors'))
