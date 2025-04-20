from safetensors import safe_open
import sys
import os
from safetensors.torch import save_file

def load_keys(filename):
    """读取safetensors文件的所有key"""
    try:
        with safe_open(filename, framework="pt") as f:  # 使用内存安全的流式读取
            return list(f.keys())
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        sys.exit(1)


def save_filtered_model(model_path, common_keys, output_path):
    """增强安全性的过滤保存函数"""
    try:
        filtered_tensors = {}
        with safe_open(model_path, framework="pt") as f:
            # 获取所有可用键集合
            available_keys = set(f.keys())

            # 按公共键过滤并加载张量
            for key in common_keys:
                if key in available_keys:  # 改用集合判断
                    filtered_tensors[key] = f.get_tensor(key)
                else:
                    print(f"\033[33m警告: 跳过不存在键 {key}\033[0m")

        # 增强路径校验
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(filtered_tensors, output_path)
        print(f"\n\033[1;32m✅ 模型已成功保存至: {output_path}\033[0m")
        print(f"原始参数数量: {len(available_keys)} → 新参数数量: {len(filtered_tensors)}")
    except Exception as e:
        print(f"\n\033[1;31m❌ 保存失败: {str(e)}\033[0m")


def analyze_keys(keys1, keys2):
    """分析两组key的异同"""
    set1, set2 = set(keys1), set(keys2)

    common = sorted(set1 & set2)  # 交集
    only_in_1 = sorted(set1 - set2)
    only_in_2 = sorted(set2 - set1)

    return common, only_in_1, only_in_2


def print_results(common, only1, only2):
    """格式化打印分析结果"""
    # 公共键输出
    print("\n\033[1;36m[公共 Keys]\033[0m (数量: {})".format(len(common)))
    print("\n".join(f"  ▸ {k}" for k in common) or "  - 无公共键 -")

    # 独占键对比
    print("\n\033[1;32m[仅存在于 Model1 的 Keys]\033[0m (数量: {})".format(len(only1)))
    print("\n".join(f"  ▶ {k}" for k in only1) or "  - 无独占键 -")

    print("\n\033[1;33m[仅存在于 Model2 的 Keys]\033[0m (数量: {})".format(len(only2)))
    print("\n".join(f"  ▶ {k}" for k in only2) or "  - 无独占键 -")


if __name__ == "__main__":
    filename = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/juliensimon/stable-diffusion-v1-5-pokemon-lora/pytorch_lora_weights.safetensors"
    filename2 = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/wO_unet/filtered_model.safetensors"
    # filename2 = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/pytorch_lora_weights_wO_unet.safetensors"

    print(f"\n\033[1;34m正在分析模型权重差异...\033[0m")
    keys1 = load_keys(filename)
    keys2 = load_keys(filename2)

    print(f"\n\033[1m模型1总key数: {len(keys1)} | 模型2总key数: {len(keys2)}\033[0m")
    print("\033[3m提示：建议关注网络层名称、参数类型（weight/bias）、模块编号的差异\033[0m")


    common, only1, only2 = analyze_keys(keys1, keys2)

    print(f"公共参数: {len(common)} | 独有参数: Model1({len(only1)}) / Model2({len(only2)})")

    print_results(common, only1, only2)


    if False:
        if len(only2) > 0:
            output_path = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/filtered_model.safetensors"
            save_filtered_model(filename2, common, output_path)
        else:
            print("\n\033[1;33m⚠️ 未检测到需要过滤的独有参数\033[0m")

