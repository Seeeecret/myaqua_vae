import os
import sys
sys.path.append("../")
import torch
from utils.models import MapperNet

def map_watermark_to_tensor(hidinfo="10101110", model_path="path_to_mapper.pt", save_path="../output/MapperWm/mapped_watermark_tensor.pth"):
    # 1. 将水印信息 "10101110" 转换为张量 (size=8)
    hid_tensor = torch.tensor([int(i) for i in hidinfo]).unsqueeze(0).float()  # 形状 (1, 8)

    # 2. 加载 MapperNet 模型
    mapper = MapperNet(input_size=len(hidinfo), output_size=8)  # 输入水印长度为8，输出大小为8
    mapper.load_state_dict(torch.load(model_path))

    # 3. 使用 MapperNet 映射水印信息
    mapped_tensor = mapper(hid_tensor)  # 获取映射后的张量



    # 5. 使用detach()去除梯度信息
    mapped_tensor = mapped_tensor.detach()

    # 4. 打印映射后的张量
    print("Mapped Watermark Tensor: ", mapped_tensor)

    # 6. 保存映射后的张量到指定文件夹（修改为 .pth 格式）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mapped_tensor, save_path)
    print(f"Mapped tensor has been saved to '{save_path}'")

# 调用函数
map_watermark_to_tensor(hidinfo="10101110", model_path="/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/output/new_rank8_8bits_6000epoch_0125/mapper.pt")

# 循环调用函数20次
for i in range(50):
    # 调整随机种子
    torch.manual_seed(i)
    map_watermark_to_tensor(hidinfo="10101110", model_path="/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/output/new_rank8_8bits_6000epoch_0125/mapper.pt", save_path=f"../output/MapperWm/loop50/mapped_watermark_tensor_{i}.pth")
