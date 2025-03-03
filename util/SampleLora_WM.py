# sample_and_save.py

import torch
import os
from safetensors.torch import load_file
import sys
sys.path.append("../")
# TODO: 导入 VAE 模型
from watermark.V3.myVAEdesign3_WatermarkV3_1 import OneDimVAEWatermark as VAE  # 导入 VAE 模型

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
    input_length = 1695744 # rank=4的aqualora
    # input_length = 797184  # 与新pokemon数据集长度相同
    # input_length = 27131904  # 与训练数据长度相同
    # input_length = 747264 # rank=4的partial
    # kld_weight = 0.005          # 与训练时相同
    kld_weight = 0.02          # 与训练时相同

    # ==============================
    # 2. 创建模型实例并加载权重
    # ==============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型实例
    model = VAE(
        latent_dim=latent_dim,
        input_length=input_length,
        kld_weight=kld_weight,
        target_fpr = 1e-6,  # +++ 新增水印参数
        lambda_w = 1.0,  # +++ 水印损失权重
        n_iters = 100  # +++ 水印优化迭代次数
    ).to(device)

    # 加载模型权重
    # TODO: 设置模型权重的路径

    # checkpoint_path = '/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/checkpoints/WMVAE_V3_alter_rank4_4bits_3000epoch_2000iters_valc/checkpoint_End/model.safetensors'
    # checkpoint_path = '/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/ckpt/WMVAE_V3_alter_rank4_4bits_3000epoch_500iters_valc/checkpoint_End/model.safetensors'
    checkpoint_path = '/baai-cwm-nas/algorithm/ziyang.yan/ckpt/WMVAE_V3_alter_rank4_4bits_3000epoch_100iters_new/checkpoint_End/model.safetensors'

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

    # 从标准正态分布中采样潜在向量 z
    generated_data = model.sample(enable_watermark=True,batch=num_samples).to(device)
    generated_data = generated_data.detach()

    # ==============================
    # 4. 保存生成的数据
    # ==============================
    # TODO: 设置保存生成数据的目录
    output_dir = '/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/generated_samples/WMVAE_V3_alter_rank4_4bits_3000epoch_100iters_new/'  # 保存生成数据的目录

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
