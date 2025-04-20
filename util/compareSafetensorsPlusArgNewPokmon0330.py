import os
import torch
from safetensors.torch import load_file
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

# 使用 argparse 定义命令行参数
parser = argparse.ArgumentParser(description="Compare safetensors files and analyze differences.")
parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory.")
parser.add_argument('--enable_sampling', type=bool, default=False, help="Enable sampling for large datasets.")
parser.add_argument('--max_samples_per_category', type=int, default=1000,
                    help="Maximum samples per category when sampling is enabled.")
parser.add_argument('--default_weight', type=str, required=True, help="Path to the default weight.")
parser.add_argument('--reconstructed_weight', type=str, required=True, help="Path to the reconstructed weight.")
args = parser.parse_args()

# 创建输出目录（如果不存在）
os.makedirs(args.output_dir, exist_ok=True)
output_dir = args.output_dir

# 加载 safetensors 文件
default_state_dict = load_file(args.default_weight, device='cuda')
reconstructed_state_dict = load_file(args.reconstructed_weight, device='cuda')
enable_sampling = args.enable_sampling
max_samples_per_category = args.max_samples_per_category


# 比较文件的结构，仅保留共同拥有的key
def compare_state_dicts(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    common_keys = keys1 & keys2
    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1

    print(f"Common keys: {len(common_keys)}")
    print(f"Only in dict1: {len(only_in_dict1)}")
    print(f"Only in dict2: {len(only_in_dict2)}")

    # 打印一些仅在其中一个文件中存在的key示例
    if only_in_dict1:
        print("\nSample keys only in default weight:")
        for key in list(only_in_dict1)[:5]:
            print(f"  {key}")

    if only_in_dict2:
        print("\nSample keys only in reconstructed weight:")
        for key in list(only_in_dict2)[:5]:
            print(f"  {key}")

    # 检查共同key的形状是否匹配
    shape_mismatch_keys = []
    for key in common_keys:
        shape1 = dict1[key].shape
        shape2 = dict2[key].shape
        if shape1 != shape2:
            shape_mismatch_keys.append(key)
            print(f"Shape mismatch for key: {key}")
            print(f"  Shape in dict1: {shape1}")
            print(f"  Shape in dict2: {shape2}")

    # 从共同key中移除形状不匹配的key
    valid_common_keys = common_keys - set(shape_mismatch_keys)

    return valid_common_keys, only_in_dict1, only_in_dict2, shape_mismatch_keys


# 比较并打印结果
common_keys, only_in_default, only_in_reconstructed, shape_mismatch_keys = compare_state_dicts(default_state_dict,
                                                                                               reconstructed_state_dict)

# 如果没有共同key，则退出程序
if not common_keys:
    print("\nError: No common keys with matching shapes found between the two models.")
    exit()

print("\nChecking common keys:")
print(f"Numbers of common keys with matching shapes: {len(common_keys)}")
for key in list(common_keys)[:5]:
    print(f"\nKey: {key}")
    print(f"  Shape in default: {default_state_dict[key].shape}")
    print(f"  Shape in reconstructed: {reconstructed_state_dict[key].shape}")

    mean_default = default_state_dict[key].mean().item()
    mean_reconstructed = reconstructed_state_dict[key].mean().item()
    std_default = default_state_dict[key].std().item()
    std_reconstructed = reconstructed_state_dict[key].std().item()

    print(f"  Mean in default: {mean_default}")
    print(f"  Mean in reconstructed: {mean_reconstructed}")
    print(f"  Std in default: {std_default}")
    print(f"  Std in reconstructed: {std_reconstructed}")

    diff = (default_state_dict[key] - reconstructed_state_dict[key]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")


# 计算 MSE Loss（仅对共同key）
def compute_mse_loss(dict1, dict2, common_keys):
    total_loss = 0.0
    count = 0
    for key in common_keys:
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        mse = F.mse_loss(tensor1, tensor2)
        total_loss += mse.item()
        count += 1

    return total_loss / count if count > 0 else 0


overall_mse_loss = compute_mse_loss(default_state_dict, reconstructed_state_dict, common_keys)
print(f"\nOverall MSE Loss between the two models (common keys only): {overall_mse_loss}")

# 定义分类规则
def_categories = {
    1: ("attentions.0", "lora.down"),
    2: ("attentions.0", "lora.up"),
    3: ("attentions.1", "lora.down"),
    4: ("attentions.1", "lora.up"),
    5: ("attentions.2", "lora.down"),
    6: ("attentions.2", "lora.up"),
}


def categorize_key(key):
    categories = def_categories
    for category, (substr1, substr2) in categories.items():
        if substr1 in key and substr2 in key:
            return category
    return None  # 如果不符合任何类别


# 分类所有共同层
category_diffs = {i: [] for i in range(1, len(def_categories) + 1)}  # 初始化添加新类别
category_cosine_sims = {i: [] for i in range(1, len(def_categories) + 1)}  # 初始化添加新类别
uncategorized_keys = []

for key in common_keys:
    category = categorize_key(key)
    if category:
        rel_diff = (default_state_dict[key] - reconstructed_state_dict[key]).abs().cpu().numpy().flatten()
        category_diffs[category].extend(rel_diff)
        # 存储余弦相似性
        cosine_sim = F.cosine_similarity(
            default_state_dict[key].view(-1),
            reconstructed_state_dict[key].view(-1),
            dim=0
        ).item()
        category_cosine_sims[category].append(cosine_sim)
    else:
        uncategorized_keys.append(key)

print(f"\nNumber of keys uncategorized: {len(uncategorized_keys)}\n")

# 定义类别标签，用于可视化，类别标签为def_categories的键值拼接，例如 "attentions.0 & lora.down"
category_labels = {i: f"{substr1} & {substr2}" for i, (substr1, substr2) in def_categories.items()}

# 打印出每个标签类别在common_keys中的数量
for i in range(1, len(def_categories) + 1):
    count = sum([categorize_key(key) == i for key in common_keys])
    print(f"Category {i} ({category_labels[i]}): {count}\n")


def plot_relative_error_distributions(category_diffs, category_labels, output_dir,
                                      enable_sampling=False, max_samples_per_category=10000):
    num_categories = len(category_diffs)
    # 动态计算网格行列数
    cols = 2  # 每行显示 2 个图
    rows = -(-num_categories // cols)  # 向上取整

    plt.figure(figsize=(5 * cols, 5 * rows))  # 动态调整画布大小
    for i in range(1, num_categories + 1):
        plt.subplot(rows, cols, i)
        data = category_diffs[i]

        # 如果启用抽样并且数据量过大，则抽样
        if enable_sampling and len(data) > max_samples_per_category:
            data = np.random.choice(data, size=max_samples_per_category, replace=False)

        if len(data) == 0:
            plt.title(f"{category_labels[i]} (No data)")
            continue

        sns.histplot(data, bins=100, kde=True, stat="density")
        plt.title(f"Relative Difference Histogram: {category_labels[i]}")
        plt.xlabel('Relative Difference')
        plt.ylabel('Density')
    plt.tight_layout()
    hist_all_path = os.path.join(output_dir, "relative_difference_histograms.png")
    plt.savefig(hist_all_path)
    plt.close()
    print(f"Relative difference histograms saved to {hist_all_path}")

    # 单独保存每个类别的相对误差分布图
    for i in range(1, num_categories + 1):
        plt.figure(figsize=(18, 6))
        data = category_diffs[i]

        # 如果启用抽样并且数据量过大，则抽样
        if enable_sampling and len(data) > max_samples_per_category:
            data = np.random.choice(data, size=max_samples_per_category, replace=False)

        if len(data) == 0:
            plt.title(f"{category_labels[i]} (No data)")
            single_cat_path = os.path.join(output_dir, f"relative_difference_histogram_category_{i}.png")
            plt.savefig(single_cat_path)
            plt.close()
            print(f"No data for category {i}, saved placeholder plot.")
            continue

        sns.histplot(data, bins=100, kde=True, stat="density")
        plt.title(f"Relative Difference Histogram: {category_labels[i]}")
        plt.xlabel('Relative Difference')
        plt.ylabel('Density')

        single_cat_path = os.path.join(output_dir, f"relative_difference_histogram_category_{i}.png")
        plt.savefig(single_cat_path)
        plt.close()
        print(f"Relative difference histogram for category {i} saved to {single_cat_path}")


# 调用绘图函数
plot_relative_error_distributions(category_diffs, category_labels, output_dir,
                                  enable_sampling=enable_sampling, max_samples_per_category=max_samples_per_category)


# 保存相对误差（可选）
def save_relative_errors(category_diffs, category_labels, output_dir, filename="relative_errors.csv"):
    data = {}
    for category, diffs in category_diffs.items():
        data[category_labels[category]] = diffs
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Relative errors saved to {csv_path}")


save_relative_errors(category_diffs, category_labels, output_dir)


# 保存未分类的键信息
def save_uncategorized_keys(uncategorized_keys, output_dir, filename="uncategorized_keys.txt"):
    if uncategorized_keys:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for key in uncategorized_keys:
                f.write(f"{key}\n")
        print(f"Uncategorized keys saved to {filepath}")
    else:
        print("No uncategorized keys to save.")


save_uncategorized_keys(uncategorized_keys, output_dir)


# 计算余弦相似性的统计量并绘制热力图
def plot_cosine_similarity_heatmap(category_cosine_sims, category_labels, output_dir):
    # 计算每个类别的平均、中位数和标准差
    cosine_stats = {
        'Category': [],
        'Mean Cosine Similarity': [],
        'Median Cosine Similarity': [],
    }

    for i in range(1, len(category_labels) + 1):
        sims = category_cosine_sims[i]
        if len(sims) > 0:
            mean_sim = np.mean(sims)
            median_sim = np.median(sims)
        else:
            mean_sim = np.nan
            median_sim = np.nan

        cosine_stats['Category'].append(category_labels[i])
        cosine_stats['Mean Cosine Similarity'].append(mean_sim)
        cosine_stats['Median Cosine Similarity'].append(median_sim)

    cosine_df = pd.DataFrame(cosine_stats)
    cosine_df.set_index('Category', inplace=True)

    # 绘制热力图
    plt.figure(figsize=(16, 8))
    sns.heatmap(cosine_df, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5, vmin=-1, vmax=1)
    plt.title('Cosine Similarity Statistics per Category')
    plt.ylabel('Category')
    plt.xlabel('Statistics')
    heatmap_path = os.path.join(output_dir, "cosine_similarity_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Cosine similarity heatmap saved to {heatmap_path}")


# 调用绘图函数
plot_cosine_similarity_heatmap(category_cosine_sims, category_labels, output_dir)


# 保存余弦相似性统计数据
def save_cosine_similarity_stats(category_cosine_sims, category_labels, output_dir,
                                 filename="cosine_similarity_stats.csv"):
    cosine_stats = {
        'Category': [],
        'Mean Cosine Similarity': [],
        'Median Cosine Similarity': [],
        'Std Cosine Similarity': []
    }

    for i in range(1, len(category_labels) + 1):
        sims = category_cosine_sims[i]
        if len(sims) > 0:
            mean_sim = np.mean(sims)
            median_sim = np.median(sims)
            std_sim = np.std(sims)
        else:
            mean_sim = np.nan
            median_sim = np.nan
            std_sim = np.nan

        cosine_stats['Category'].append(category_labels[i])
        cosine_stats['Mean Cosine Similarity'].append(mean_sim)
        cosine_stats['Median Cosine Similarity'].append(median_sim)
        cosine_stats['Std Cosine Similarity'].append(std_sim)

    cosine_df = pd.DataFrame(cosine_stats)
    csv_path = os.path.join(output_dir, filename)
    cosine_df.to_csv(csv_path, index=False)
    print(f"Cosine similarity statistics saved to {csv_path}")


save_cosine_similarity_stats(category_cosine_sims, category_labels, output_dir)

print("Analysis completed successfully.")