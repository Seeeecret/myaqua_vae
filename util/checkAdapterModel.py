import torch
from safetensors.torch import load_file


def load_and_flatten_weights(filename, file_type='pth'):
    """
    根据 file_type 参数，选择读取 .pth 或 .safetensors 文件，并展平所有权重。

    :param filename: 模型文件路径
    :param file_type: 选择 'pth' 或 'safetensors'
    :return: (flattened_weights, total_weights)
             其中 flattened_weights 是一个字典，key 为参数名，value 为展平后的张量；
             total_weights 是所有张量元素数量的总和。
    """
    # 1. 根据 file_type 加载权重
    if file_type == 'pth':
        # 读取 .pth 文件（可以加上 map_location 防止在无 GPU 环境下读取 CUDA 权重报错）
        model = torch.load(filename, map_location='cpu')

        if "state_dict" in model:
            # 如果读取出来的 model 是一个包含 state_dict 的字典
            state_dict = model["state_dict"]
        else:
            # 如果读取出来本身就是 state_dict
            state_dict = model

    elif file_type == 'safetensors':
        # 读取 .safetensors 文件
        state_dict = load_file(filename)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # 2. 遍历并展平每个权重
    flattened_weights = {}
    for key, value in state_dict.items():
        print(f"Key: {key}, Original Shape: {value.shape}")
        # 展平成一维张量
        flattened_value = value.flatten()
        flattened_weights[key] = flattened_value
        print(f"Key: {key}, Flattened Shape: {flattened_value.shape}")

    # 3. 计算所有张量元素数量之和
    total_weights = sum(tensor.numel() for tensor in flattened_weights.values())
    print(f"Total weights across all tensors: {total_weights}")

    return flattened_weights, total_weights


filename_pth = "/data/Tsinghua/wuzy/rank8_bits8_dataset/normalized_partial_data_9360/normalized_pytorch_lora_weights_11.pth"
flattened_pth, total_pth = load_and_flatten_weights(filename_pth, file_type='pth')

