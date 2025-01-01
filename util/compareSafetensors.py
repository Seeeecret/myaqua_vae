import torch
from safetensors.torch import load_file
import torch.nn.functional as F

# 定义文件路径
file_default = '/mnt/share_disk/dorin/AquaLoRA/train/rank64_output/default_rank64/pytorch_lora_weights.safetensors'
file_default_step37432 = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/pytorch_lora_weights_37432.safetensors'
file_default_step37499 = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/pytorch_lora_weights_37499.safetensors'

SHAO_default = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/bus/adapter_model_31.safetensors'
rank64_sample_1_1202 = '../output/rank64_alter3_kld_weight_0005_1202/pytorch_lora_weights.safetensors'
rank64_sample_1_1204 = '../output/rank64_alter3_kld_weight_0005_1202/pytorch_lora_weights.safetensors'

encodeDecodeData_path = '../output/encodedecode_rank64_alter3_kld_weight_0005_1205/pytorch_lora_weights.safetensors'

rank4_file_default_step37499 = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank4_extracted_lora_weights/pytorch_lora_weights_37499.safetensors'
rank8_8bits_default = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights/pytorch_lora_weights_18750.safetensors'
rank16_8bits_default = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank16_8bits_extracted_lora_weights/pytorch_lora_weights_18750.safetensors'

file_reconstructed = '/mnt/share_disk/dorin/AquaLoRA/train/rank64_output/reconstructed_rank64/pytorch_lora_weights.safetensors'
file_reconstructed_2 = '/mnt/share_disk/dorin/AquaLoRA/output/rank64_alter3_kld_weight_0005/pytorch_lora_weights.safetensors'

SHAO_reconstructed = '/mnt/share_disk/dorin/AquaLoRA/output/SHAO_alter3_kld_weight_0005_1208/adapter_model.safetensors'
SHAO_reconstructed_1209 = '/mnt/share_disk/dorin/AquaLoRA/output/SHAO_alter3_kld_weight_0005_1209/adapter_model.safetensors'

encodeDecodeData_SHAO_path = '/mnt/share_disk/dorin/AquaLoRA/output/encodedecode_SHAO_alter3_kld_weight_0005_1208/adapter_model.safetensors'
encodeDecodeData_SHAO_path_NoNor_1209 = '/mnt/share_disk/dorin/AquaLoRA/output/encodedecode_SHAO_alter3_kld_weight_0005_1209/adapter_model.safetensors'

rank4_file_reconstructed = '/mnt/share_disk/dorin/AquaLoRA/output/rank4_alter3_kld_weight_0005/pytorch_lora_weights.safetensors'

rank8_8bits_file_8000epoch_sample = '/mnt/share_disk/dorin/AquaLoRA/output/rank8_8bits_lora_vae_checkpoints_1216_8000epoch/pytorch_lora_weights.safetensors'
rank8_8bits_partial_file_8000epoch_sample = '/mnt/share_disk/dorin/AquaLoRA/output/rank8_8bits_partial_lora_vae_checkpoints_1219_8000epoch/completed/pytorch_lora_weights.safetensors'

rank16_8bits_file_800epoch_sample = '/mnt/share_disk/dorin/AquaLoRA/output/rank16_8bits_lora_vae_checkpoints_1214/pytorch_lora_weights.safetensors'


test_reconstructed_step37432 = '../output/rank64_alter3_kld_weight_00005/test_pytorch_lora_weights.safetensors'

# TODO: 选择要加载的两个 safetensors 文件
default_state_dict = load_file(rank8_8bits_default, device='cuda')
reconstructed_state_dict = load_file(rank8_8bits_partial_file_8000epoch_sample, device='cuda')

# 比较文件的结构
def compare_state_dicts(dict1, dict2):
    # 获取所有的键
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # 比较键的差异
    common_keys = keys1 & keys2
    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1

    print(f"Common keys: {len(common_keys)}")
    print(f"Only in dict1: {len(only_in_dict1)}")
    print(f"Only in dict2: {len(only_in_dict2)}")

    # 比较每个键对应的张量形状
    for key in common_keys:
        shape1 = dict1[key].shape
        shape2 = dict2[key].shape
        if shape1 != shape2:
            print(f"Shape mismatch for key: {key}")
            print(f"  Shape in dict1: {shape1}")
            print(f"  Shape in dict2: {shape2}")

    return common_keys, only_in_dict1, only_in_dict2

# 比较并打印结果
common_keys, only_in_default, only_in_reconstructed = compare_state_dicts(default_state_dict, reconstructed_state_dict)

# 输出常见键的详细信息
print("\nChecking common keys:")
for key in common_keys:
    print(f"\nKey: {key}")
    print(f"  Shape in default: {default_state_dict[key].shape}")
    print(f"  Shape in reconstructed: {reconstructed_state_dict[key].shape}")

    # 计算并输出均值和标准差
    mean_default = default_state_dict[key].mean().item()
    mean_reconstructed = reconstructed_state_dict[key].mean().item()
    std_default = default_state_dict[key].std().item()
    std_reconstructed = reconstructed_state_dict[key].std().item()

    print(f"  Mean in default: {mean_default}")
    print(f"  Mean in reconstructed: {mean_reconstructed}")
    print(f"  Std in default: {std_default}")
    print(f"  Std in reconstructed: {std_reconstructed}")

    # 计算元素级的差异
    diff = (default_state_dict[key] - reconstructed_state_dict[key]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")

# 计算两个文件整体的 MSE Loss
def compute_mse_loss(dict1, dict2):
    total_loss = 0.0
    for key in dict1.keys():
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        mse = F.mse_loss(tensor1, tensor2)
        total_loss += mse.item()

    return total_loss / len(dict1) if len(dict1) > 0 else 0
def compute_differences(tensor_a, tensor_b):
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / (torch.abs(tensor_b) + 1e-8)  # 避免除以零
    cosine_sim = F.cosine_similarity(tensor_a.view(-1), tensor_b.view(-1), dim=0)
    l1_loss = torch.mean(abs_diff)
    mse_loss = F.mse_loss(tensor_a, tensor_b)
    return {
        'absolute_difference': abs_diff,
        'relative_difference': rel_diff,
        'cosine_similarity': cosine_sim.item(),
        'l1_loss': l1_loss.item(),
        'mse_loss': mse_loss.item()
    }

# MSE Loss
overall_mse_loss = compute_mse_loss(default_state_dict, reconstructed_state_dict)

print(f"\nOverall MSE Loss between the two models: {overall_mse_loss}")

# 如果存在只在其中一个文件中的键，也可以输出它们的形状
if only_in_default:
    print("\nKeys only in the default file:")
    for key in only_in_default:
        print(f"  {key}: {default_state_dict[key].shape}")

if only_in_reconstructed:
    print("\nKeys only in the reconstructed file:")
    for key in only_in_reconstructed:
        print(f"  {key}: {reconstructed_state_dict[key].shape}")
