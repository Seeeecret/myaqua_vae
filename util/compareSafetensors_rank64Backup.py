import torch
from safetensors.torch import load_file

# 定义文件路径
file_default = '/mnt/share_disk/dorin/AquaLoRA/train/rank64_output/default_rank64/pytorch_lora_weights.safetensors'
file_default_step37432 = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/pytorch_lora_weights_37432.safetensors'
file_default_step37499 = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/pytorch_lora_weights_37499.safetensors'

rank4_file_default_step37499 = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank4_extracted_lora_weights/pytorch_lora_weights_37499.safetensors'


file_reconstructed = '/mnt/share_disk/dorin/AquaLoRA/train/rank64_output/reconstructed_rank64/pytorch_lora_weights.safetensors'
file_reconstructed_2 = '/mnt/share_disk/dorin/AquaLoRA/output/rank64_alter3_kld_weight_0005/pytorch_lora_weights.safetensors'
rank4_file_reconstructed = '/mnt/share_disk/dorin/AquaLoRA/output/rank4_alter3_kld_weight_0005/pytorch_lora_weights.safetensors'

test_reconstructed_step37432 = '../output/rank64_alter3_kld_weight_00005/test_pytorch_lora_weights.safetensors'
# 加载两个 safetensors 文件
default_state_dict = load_file(file_default_step37499, device='cuda')
reconstructed_state_dict = load_file(file_reconstructed_2, device='cuda')

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
    print(f"  Mean in default: {default_state_dict[key].mean().item()}")
    print(f"  Mean in reconstructed: {reconstructed_state_dict[key].mean().item()}")
    print(f"  Std in default: {default_state_dict[key].std().item()}")
    print(f"  Std in reconstructed: {reconstructed_state_dict[key].std().item()}")

# 如果存在只在其中一个文件中的键，也可以输出它们的形状
if only_in_default:
    print("\nKeys only in the default file:")
    for key in only_in_default:
        print(f"  {key}: {default_state_dict[key].shape}")

if only_in_reconstructed:
    print("\nKeys only in the reconstructed file:")
    for key in only_in_reconstructed:
        print(f"  {key}: {reconstructed_state_dict[key].shape}")
