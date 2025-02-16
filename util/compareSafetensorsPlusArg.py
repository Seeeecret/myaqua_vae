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
parser.add_argument('--max_samples_per_category', type=int, default=1000, help="Maximum samples per category when sampling is enabled.")
parser.add_argument('--default_weight', type=str, required=True, help="Path to the default weight.")
parser.add_argument('--reconstructed_weight', type=str, required=True, help="Path to the reconstructed weight.")
args = parser.parse_args()

# 创建输出目录（如果不存在）
os.makedirs(args.output_dir, exist_ok=True)
output_dir = args.output_dir
# 加载 safetensors 文件
default_state_dict = load_file(args.default_weight, device='cuda')
reconstructed_state_dict = load_file(args.reconstructed_weight, device='cuda')

# 比较文件的结构
def compare_state_dicts(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    common_keys = keys1 & keys2
    only_in_dict1 = keys1 - keys2
    only_in_dict2 = keys2 - keys1

    print(f"Common keys: {len(common_keys)}")
    print(f"Only in dict1: {len(only_in_dict1)}")
    print(f"Only in dict2: {len(only_in_dict2)}")

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

print("\nChecking common keys:")
print(f"Numbers of common keys: {len(common_keys)}")
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

# 计算 MSE Loss
def compute_mse_loss(dict1, dict2):
    total_loss = 0.0
    for key in dict1.keys():
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        mse = F.mse_loss(tensor1, tensor2)
        total_loss += mse.item()

    return total_loss / len(dict1) if len(dict1) > 0 else 0

overall_mse_loss = compute_mse_loss(default_state_dict, reconstructed_state_dict)
print(f"\nOverall MSE Loss between the two models: {overall_mse_loss}")

# 逐元素差异版
def compute_differences(tensor_a, tensor_b):
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / (torch.abs(tensor_b))  # 避免除以零
    # rel_diff = abs_diff / (torch.abs(tensor_b) + 1e-8)  # 避免除以零
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
# 逐Key计算版本
def compute_differencesV2(tensor_a, tensor_b):
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / (torch.abs(tensor_b))  # 避免除以零
    cosine_sim = F.cosine_similarity(tensor_a.view(-1), tensor_b.view(-1), dim=0)
    l1_loss = torch.mean(abs_diff)
    mse_loss = F.mse_loss(tensor_a, tensor_b)
    return {
        'absolute_difference': torch.mean(abs_diff),
        'relative_difference': torch.mean(rel_diff),
        'cosine_similarity': (cosine_sim.item()),
        'l1_loss': l1_loss.item(),
        'mse_loss': mse_loss.item()
    }


# 计算所有层的差异并存储
diffs = {}
for key in common_keys:
    diffs[key] = compute_differencesV2(reconstructed_state_dict[key], default_state_dict[key])

# 计算整体 MSE Loss
overall_mse_loss = compute_mse_loss(default_state_dict, reconstructed_state_dict)
print(f"\nOverall MSE Loss between the two models: {overall_mse_loss}")

# 输出仅存在于一个模型中的键（限制为前5个键）
if only_in_default:
    print("\nKeys only in the default file:")
    for key in list(only_in_default)[:5]:
        print(f"  {key}: {default_state_dict[key].shape}")

if only_in_reconstructed:
    print("\nKeys only in the reconstructed file:")
    for key in list(only_in_reconstructed)[:5]:
        print(f"  {key}: {reconstructed_state_dict[key].shape}")


# 定义分类规则
def categorize_key(key):
    categories = {
        1: ("ff", "down.weight"),
        2: ("ff", "up.weight"),
        3: ("attn", "down.weight"),
        4: ("attn", "up.weight"),
        5: ("proj_in", "down.weight"),
        6: ("proj_in", "up.weight"),
        7: ("proj_out", "down.weight"),
        8: ("proj_out", "up.weight")
    }
    for category, (substr1, substr2) in categories.items():
        if substr1 in key and substr2 in key:
            return category
    return None  # 如果不符合任何类别


# 分类所有层
category_diffs = {i: [] for i in range(1, 9)}  # 初始化8个类别
category_cosine_sims = {i: [] for i in range(1, 9)}  # 初始化8个类别的余弦相似性
uncategorized_keys = []

for key in common_keys:
    category = categorize_key(key)
    if category:
        rel_diff = diffs[key]['relative_difference'].cpu().numpy().flatten()
        category_diffs[category].extend(rel_diff)
        # 存储余弦相似性
        category_cosine_sims[category].append(diffs[key]['cosine_similarity'])
    else:
        uncategorized_keys.append(key)

print(f"\nNumber of keys uncategorized: {len(uncategorized_keys)}\n")

# 定义类别标签
category_labels = {
    1: "FF & Down.weight",
    2: "FF & Up.weight",
    3: "Attn & Down.weight",
    4: "Attn & Up.weight",
    5: "Proj_in & Down.weight",
    6: "Proj_in & Up.weight",
    7: "Proj_out & Down.weight",
    8: "Proj_out & Up.weight"
}

# 打印出每个标签类别在common_keys中的数量
for i in range(1, 9):
    count = sum([categorize_key(key) == i for key in common_keys])
    print(f"Category {i} ({category_labels[i]}): {count}\n")


# 可视化相对误差分布并保存图片
# def plot_relative_error_distributions(category_diffs, category_labels, output_dir,
#                                       enable_sampling=False, max_samples_per_category=10000):
#     num_categories = len(category_diffs)
#     plt.figure(figsize=(20, 20))
#     for i in range(1, num_categories + 1):
#         plt.subplot(4, 2, i)
#         data = category_diffs[i]
#
#         # 如果启用抽样并且数据量过大，则抽样
#         if enable_sampling and len(data) > max_samples_per_category:
#             data = np.random.choice(data, size=max_samples_per_category, replace=False)
#
#         if len(data) == 0:
#             plt.title(f"{category_labels[i]} (No data)")
#             continue
#
#         sns.histplot(data, bins=100, kde=True, stat="density")
#         plt.title(f"Relative Difference Histogram: {category_labels[i]}")
#         # from matplotlib.ticker import MaxNLocator
#         # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=50))  # 自动调整横轴刻度为 20 个
#
#         plt.xlabel('Relative Difference')
#         plt.ylabel('Density')
#     plt.tight_layout()
#     hist_all_path = os.path.join(output_dir, "relative_difference_histograms.png")
#     plt.savefig(hist_all_path)
#     plt.close()
#     print(f"Relative difference histograms saved to {hist_all_path}")
#
#     # 单独保存每个类别的相对误差分布图
#     for i in range(1, num_categories + 1):
#         plt.figure(figsize=(18, 6))
#         data = category_diffs[i]
#
#         # 如果启用抽样并且数据量过大，则抽样
#         if enable_sampling and len(data) > max_samples_per_category:
#             data = np.random.choice(data, size=max_samples_per_category, replace=False)
#
#         if len(data) == 0:
#             plt.title(f"{category_labels[i]} (No data)")
#             single_cat_path = os.path.join(output_dir, f"relative_difference_histogram_category_{i}.png")
#             plt.savefig(single_cat_path)
#             plt.close()
#             print(f"No data for category {i}, saved placeholder plot.")
#             continue
#
#         sns.histplot(data, bins=100, kde=True, stat="density")
#         plt.title(f"Relative Difference Histogram: {category_labels[i]}")
#         # from matplotlib.ticker import MaxNLocator
#         # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=50))  # 自动调整横轴刻度为 20 个
#
#         plt.xlabel('Relative Difference')
#         plt.ylabel('Density')
#
#         single_cat_path = os.path.join(output_dir, f"relative_difference_histogram_category_{i}.png")
#         plt.savefig(single_cat_path)
#         plt.close()
#         print(f"Relative difference histogram for category {i} saved to {single_cat_path}")
def plot_relative_error_distributions(category_diffs, category_labels, output_dir,
                                      enable_sampling=False, max_samples_per_category=10000):
    num_categories = len(category_diffs)
    # 动态计算网格行列数
    cols = 2  # 每行显示 3 个图
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
                                  enable_sampling=args.enable_sampling, max_samples_per_category=args.max_samples_per_category)


# 如果需要，也可以将相对误差保存为CSV或其他格式
def save_relative_errors(category_diffs, category_labels, output_dir, filename="relative_errors.csv"):
    data = {}
    for category, diffs in category_diffs.items():
        data[category_labels[category]] = diffs
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"Relative errors saved to {csv_path}")


# 保存相对误差（可选）
save_relative_errors(category_diffs, category_labels, output_dir)


# 如果需要，可以保存未分类的键信息
def save_uncategorized_keys(uncategorized_keys, output_dir, filename="uncategorized_keys.txt"):
    if uncategorized_keys:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for key in uncategorized_keys:
                f.write(f"{key}\n")
        print(f"Uncategorized keys saved to {filepath}")
    else:
        print("No uncategorized keys to save.")


# 计算余弦相似性的统计量并绘制热力图
def plot_cosine_similarity_heatmap(category_cosine_sims, category_labels, output_dir):
    # 计算每个类别的平均、中位数和标准差
    cosine_stats = {
        'Category': [],
        'Mean Cosine Similarity': [],
        'Median Cosine Similarity': [],
        # 'Std Cosine Similarity': []
    }

    for i in range(1, 9):
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
        # cosine_stats['Std Cosine Similarity'].append(std_sim)

    cosine_df = pd.DataFrame(cosine_stats)
    cosine_df.set_index('Category', inplace=True)

    # 绘制热力图
    plt.figure(figsize=(16, 8))
    sns.heatmap(cosine_df, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5,vmin=-1, vmax=1)
    plt.title('Cosine Similarity Statistics per Category')
    plt.ylabel('Category')
    plt.xlabel('Statistics')
    heatmap_path = os.path.join(output_dir, "cosine_similarity_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Cosine similarity heatmap saved to {heatmap_path}")


# 调用绘图函数
plot_cosine_similarity_heatmap(category_cosine_sims, category_labels, output_dir)


# 也可以将余弦相似性统计数据保存为 CSV
def save_cosine_similarity_stats(category_cosine_sims, category_labels, output_dir,
                                 filename="cosine_similarity_stats.csv"):
    cosine_stats = {
        'Category': [],
        'Mean Cosine Similarity': [],
        'Median Cosine Similarity': [],
        'Std Cosine Similarity': []
    }

    for i in range(1, 9):
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


# 保存余弦相似性统计数据
save_cosine_similarity_stats(category_cosine_sims, category_labels, output_dir)

print("End")
