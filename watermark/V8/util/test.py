import torch
import torch.nn.functional as F

# 创建一个示例tensor
msg = torch.arange(64).reshape(1, 1, 8, 8).float()
print("原始tensor:")
print(msg.squeeze())

# 创建旋转后的版本
rot90 = torch.rot90(msg, k=1, dims=[2, 3])
rot180 = torch.rot90(msg, k=2, dims=[2, 3])
rot270 = torch.rot90(msg, k=3, dims=[2, 3])

# 打印旋转后的结果
print("\n旋转90度:")
print(rot90.squeeze())
print("\n旋转180度:")
print(rot180.squeeze())
print("\n旋转270度:")
print(rot270.squeeze())

# 拼接所有tensor
result = torch.cat([msg, rot90, rot180, rot270], dim=1)
print("\n最终拼接结果 shape:", result.shape)
print("最终tensor的四个通道:")
for i in range(4):
    print(f"\n通道 {i}:")
    print(result[0, i])
