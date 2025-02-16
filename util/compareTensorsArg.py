import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 解析命令行参数
parser = argparse.ArgumentParser(description="Compare two tensors from .pth or .pt files.")
parser.add_argument('--tensor_a', type=str, required=True, help="Path to the ground truth tensor (A).")
parser.add_argument('--tensor_b', type=str, required=True, help="Path to the generated tensor (B).")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the results.")
parser.add_argument('--save_b_with_grad', action="store_true", help="Save tensor B with gradient tracking.")
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

# 读取tensor
tensor_a = torch.load(args.tensor_a, map_location="cpu")  # Ground truth
tensor_b = torch.load(args.tensor_b, map_location="cpu")  # Generated tensor

# **确保 tensorB 具有与 tensorA 相同的形状**
tensor_b = tensor_b.view(1, -1)  # 转换为 [1, 8] 形状

# 确保两个 tensor 维度相同
if tensor_a.shape != tensor_b.shape:
    raise ValueError(f"Shape mismatch: A {tensor_a.shape} vs B {tensor_b.shape}")

# **计算整体余弦相似度**
cosine_similarities = F.cosine_similarity(tensor_a.flatten(), tensor_b.flatten(), dim=0).item()

# **计算逐元素相对误差**
relative_errors = torch.abs((tensor_a - tensor_b) / (tensor_a + 1e-8))  # 避免除零错误
relative_errors = relative_errors.numpy().flatten()  # 保持 1D 形状

# 打印结果
print("\nCosine Similarity (Single Value):")
print(cosine_similarities)

print("\nRelative Errors (Element-wise):")
print(relative_errors)

# **修正 DataFrame**
df = pd.DataFrame({
    "Index": np.arange(tensor_a.shape[1]),  # 生成索引
    "Relative Error": relative_errors  # 1D 数组
})

# 单独存储整体余弦相似度
cosine_df = pd.DataFrame({
    "Cosine Similarity": [cosine_similarities]
})

# **保存 CSV**
csv_path = os.path.join(args.output_dir, "tensor_comparison_results.csv")
df.to_csv(csv_path, index=False)
print(f"Relative Errors saved to {csv_path}")

cosine_csv_path = os.path.join(args.output_dir, "cosine_similarity.csv")
cosine_df.to_csv(cosine_csv_path, index=False)
print(f"Cosine Similarity saved to {cosine_csv_path}")

# **可视化**
plt.figure(figsize=(10, 5))

# 相对误差分布
plt.subplot(1, 1, 1)
sns.barplot(x=df["Index"], y=df["Relative Error"])
plt.title("Relative Error per Element")
plt.xlabel("Element Index")
plt.ylabel("Relative Error")

# 保存图像
plt.tight_layout()
plot_path = os.path.join(args.output_dir, "tensor_comparison_plot.png")
plt.savefig(plot_path)
plt.show()

print(f"Comparison plot saved to {plot_path}")

# **将 tensorB 转换为带有计算图的格式，并保存**
if args.save_b_with_grad:
    # UserWarning: To copy construct from a tensor,
    # it is recommended to use sourceTensor.clone().detach()
    # or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    tensor_b_with_grad = tensor_b.clone().detach().requires_grad_(True)
    # tensor_b_with_grad = torch.tensor(tensor_b, requires_grad=True)  # 使 tensorB 具有梯度跟踪
    save_path = os.path.join(args.output_dir, "tensor_b_with_grad.pth")
    torch.save(tensor_b_with_grad, save_path)
    print(f"Tensor B: {tensor_b_with_grad}")
    print(f"Tensor B with grad_fn saved to {save_path}")
