from safetensors.torch import load_file

filename = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/bus/adapter_model_14.safetensors"
model = load_file(filename)
state_dict = model
# for key, value in model.items():
#     print(key, value)

# 遍历并展平每个权重
flattened_weights = {}
for key, value in model.items():
    print(f"Key: {key}, Original Shape: {value.shape}")
    # 展平成一维张量
    flattened_weights[key] = value.view(-1)  # 或使用 value.flatten()
    print(f"Key: {key}, Flattened Shape: {flattened_weights[key].shape}")

# 如果需要进一步处理或保存展平后的权重
# 示例：计算总权重数目
total_weights = sum(tensor.numel() for tensor in flattened_weights.values())
print(f"Total weights across all tensors: {total_weights}")
