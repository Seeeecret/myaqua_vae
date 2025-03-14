# sample_and_save.py

import torch
import os
from safetensors.torch import load_file
import sys
sys.path.append("../")
# TODO: 导入 VAE 模型
# from myVAEdesign3_rank8_partial import OneDimVAE as VAE  # 导入 VAE 模型
# from myVAEdesign3_rank4 import OneDimVAE as VAE  # 导入 VAE 模型

# from myVAEdesign3_rank4_0110 import OneDimVAE as VAE

# from myVAEdesign3_rank16 import OneDimVAE as VAE  # 导入 VAE 模型
# from myVAEdesign3_SHAO import OneDimVAE as VAE  # 导入 VAE 模型
# from myVAEdesign3_rank8_partial import OneDimVAE as VAE  # 导入 VAE 模型
# from myVAEdesign3_rank64 import OneDimVAE as VAE  # 导入 VAE 模型
from myVAEdesign3_WmV import OneDimVAE as VAE  # 导入 VAE 模型

def main():
    # ==============================
    # 1. 设置模型参数
    # ==============================
    # TODO: 设置模型参数
    latent_dim = 256            # 与训练时相同
    # latent_dim = 4096            # 与训练时相同
    # input_length = 1494528  # 与训练数据长度相同
    # input_length = 3391488  # AQUALORA RANK=8
    # input_length = 1494528  # AQUALORA RANK=8 partial
    # input_length = 6782976  # 与训练数据长度相同
    input_length = 1695744
    # input_length = 797184  # 与新pokemon数据集长度相同
    # input_length = 27131904  # 与训练数据长度相同
    # input_length = 8 # rank=4
    # input_length = 747264 # rank=4的partial
    # kld_weight = 0.005          # 与训练时相同
    kld_weight = 0.02          # 与训练时相同
    iters = 500

    # ==============================
    # 2. 创建模型实例并加载权重
    # ==============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型实例
    model = VAE(
        input_length=args.input_length,
        latent_dim=args.latent_dim,
        kld_weight=args.kld_weight,
        target_fpr=1e-6,  # +++ 新增水印参数
        lambda_w=1.0,  # +++ 水印损失权重
        n_iters=iters  # +++ 水印优化迭代次数
    ).to(device)

    # 加载模型权重
    # TODO: 设置模型权重的路径
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_lora_vae_checkpoints_1216_8000epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_lora_vae_checkpoints_1216_4000epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_partial_lora_vae_checkpoints_1219_8000epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_lora_vae_checkpoints_1225_150epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_pokemon_3000epoch_latD1024_0120/checkpoint_End_CVAE/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_pokemon_2000epoch_0120/checkpoint_End_CVAE/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_pokemon_3000epoch_latD4096_0123/checkpoint_End_CVAE/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_rank4_4bits_8000epoch_0202/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/WmV2_rank8_8bits_3000epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/WmV2_rank8_8bits_3000epoch/checkpoint_End/model.safetensors'
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/better_rank4_4bits_partial_6000epoch_0204/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_rank8_8bits_6000epoch_partial_0201/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_rank8_8bits_10000epoch_0125/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/new_rank8_8bits_6000epoch_0125/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/YMX_dog_r8_voc_2000epoch_0116/checkpoint_End_CVAE/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_lora_vae_checkpoints_1225_150epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_partial_lora_vae_checkpoints_1219_150epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_partial_lora_vae_checkpoints_1219_800epoch/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank8_8bits_lora_vae_checkpoints_1215/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank16_8bits_lora_vae_checkpoints_1214/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/SHAO_lora_vae_checkpoints_1209/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/SHAO_lora_vae_checkpoints_1208/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1204/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1202/checkpoint_epoch_400/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1127/checkpoint_epoch_400/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_vae_checkpoints_alter3/checkpoint_epoch_280/model.safetensors'  # 根据实际路径修改
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 加载权重到模型中
    model.load_state_dict(load_file(checkpoint_path))
    print("Model weights loaded successfully.")

    # 设置模型为评估模式
    model.eval()

    # ==============================
    # 3. 采样生成数据
    # ==============================
    num_samples = 1  # 生成样本的数量
    with torch.no_grad():
        # 从标准正态分布中采样潜在向量 z
        generated_data = model.sample(num_samples).to(device)

    # ==============================
    # 4. 保存生成的数据
    # ==============================
    # TODO: 设置保存生成数据的目录
    # output_dir = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch/20241216'  # 保存生成数据的目录
    output_dir = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/generated_samples/WmV2_rank8_8bits_3000epoch/'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank8_8bits_kld_weight_0005_150epoch/20241225'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank8_8bits_kld_weight_0005_8000epoch_partial/20241219'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank8_8bits_kld_weight_0005_800epoch_partial/20241219'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank8_8bits_kld_weight_0005_4000epoch/20241216'  # 保存生成数据的目录
    # output_dir = '../generated_samples/SHAO_kld_weight_0005/20241209'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank64_kld_weight_0005/20241204'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank64_kld_weight_0005/20241202'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank4_kld_weight_0005/20241122'  # 保存生成数据的目录
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        sample = generated_data[i].cpu()
        sample_path = os.path.join(output_dir, f'sample_{i + 1}.pth')
        sample = sample[:input_length]
        torch.save(sample, sample_path)
        print(f"Sample {i + 1} saved to {sample_path}")
        print(f"Sample shape: {sample.shape}")
        print("Sample: ", sample)

    print("All samples generated and saved successfully.")

if __name__ == '__main__':
    main()
