import os
import argparse
import torch
import safetensors
from safetensors.torch import save_file
import sys
from TrainFreeMapper import TrainFreeMapper

sys.path.append("../")


def create_watermark_lora_free(train_folder, scale, msg_bits=48, hidinfo=None,
                               output_size=None, std=1.0):
    """
    使用TrainFreeMapper的无训练水印嵌入版本

    参数变化：
    新增std参数控制映射幅度
    移除与训练相关的参数（use_sampleMsgVector等）
    """
    # 加载原始LoRA权重
    lora_path = f"{train_folder}/pytorch_lora_weights.safetensors"
    lora_state_dict = safetensors.torch.load_file(lora_path, device='cuda')

    # 生成/验证水印信息
    if hidinfo is None:
        hidinfo = torch.randint(0, 2, (1, msg_bits))
    else:
        assert len(hidinfo) == msg_bits, f"水印长度需为{msg_bits}位，实际得到{len(hidinfo)}"
        hidinfo = torch.tensor([int(b) for b in hidinfo]).unsqueeze(0)

    # 初始化无训练映射器
    mapper = TrainFreeMapper(input_size=msg_bits, output_size=output_size, std=std)
    # 生成映射向量（无需加载任何训练参数）
    mapped_loradiag = mapper(hidinfo.float()).cuda()

    # 修改LoRA权重（与原逻辑相同）
    c_lora_state_dict = {}
    for key in lora_state_dict:
        if 'unet' in key:
            # 处理注意力层和FF层
            if 'attn' in key or 'ff' in key:
                if 'up.weight' in key:
                    c_lora_state_dict[key] = lora_state_dict[key]
                elif 'down.weight' in key:
                    mid = torch.diag_embed(mapped_loradiag)[0]  # [D, D]
                    c_lora_state_dict[key] = mid @ lora_state_dict[key] * scale

            # 处理投影层
            if 'proj_in' in key or 'proj_out' in key:
                if 'up.weight' in key:
                    c_lora_state_dict[key] = lora_state_dict[key]
                elif 'down.weight' in key:
                    mid = mapped_loradiag[0]  # [D]
                    # 广播到卷积权重形状 [C_out, C_in, K, K]
                    c_lora_state_dict[key] = lora_state_dict[key] * mid[:, None, None, None] * scale
        elif 'text_encoder' in key:
            continue  # 跳过文本编码器
        else:
            raise ValueError(f"未知的权重键名: {key}")

    # 生成二进制字符串表示
    hidinfo_str = ''.join(map(str, hidinfo.squeeze().tolist()))

    # 保存路径管理
    save_dir = f"{train_folder}/free_{hidinfo_str}"
    os.makedirs(save_dir, exist_ok=True)

    # 保存修改后的权重
    save_path = f"{save_dir}/pytorch_lora_weights.safetensors"
    save_file(c_lora_state_dict, save_path)

    return hidinfo_str, c_lora_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="无训练水印嵌入器")
    parser.add_argument("--train_folder", type=str, required=True,
                        help="原始LoRA权重目录")
    parser.add_argument("--msg_bits", type=int, default=48,
                        help="水印信息位数")
    parser.add_argument("--scale", type=float, default=1.03,
                        help="水印缩放因子")
    parser.add_argument("--hidinfo", type=str, default=None,
                        help="自定义水印二进制字符串，如'0101...'")
    parser.add_argument("--output_size", type=int, default=None,
                        help="映射输出维度（默认等于msg_bits）")
    parser.add_argument("--std", type=float, default=1.0,
                        help="映射幅度控制参数")

    args = parser.parse_args()

    output_size = args.output_size
    # 设置输出维度
    if output_size is None:
        output_size = args.msg_bits
    # 执行水印嵌入
    hidinfo, _ = create_watermark_lora_free(
        train_folder=args.train_folder,
        scale=args.scale,
        msg_bits=args.msg_bits,
        hidinfo=args.hidinfo,
        output_size=output_size,
        std=args.std
    )

    print(f"生成水印标识: {hidinfo}")
    print(f"带水印的LoRA权重已保存至: {args.train_folder}/free_{hidinfo}/")
