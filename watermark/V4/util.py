from typing import Tuple, Any, Union
from scipy import special
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import scipy.special


def _cosine_similarity_pvalue(
        cosine_sim: torch.Tensor,  # 余弦相似度 [B]
        d: int  # 潜在空间维度
) -> torch.Tensor:
    """
    计算随机单位向量投影的p值
    公式：P(|u·a| >= c) = 1 - I_{c^2}( (d-1)/2, 0.5 )
    其中 I 是正则化不完全Beta函数
    """
    c_sq = cosine_sim ** 2  # [B]
    a = (d - 1) / 2.0
    b = 0.5

    # 利用Scipy的betainc函数
    dtype_origin = c_sq.dtype
    device_origin = c_sq.device  # 保存原始设备
    p_value = torch.ones_like(c_sq, dtype=torch.float64)
    valid_mask = (c_sq < 1.0)  # c=1时p=0，但需要处理数值误差

    # 计算Beta分布的不完全函数
    c_sq_numpy = c_sq[valid_mask].detach().cpu().numpy() # 转换为numpy数组
    p_value_numpy = 1.0 - scipy.special.betainc(a, b, c_sq_numpy)  # 使用Scipy的betainc计算

    # 将结果转回torch Tensor并移回原始设备
    p_value[valid_mask] = torch.from_numpy(p_value_numpy).to(dtype=torch.float64, device=device_origin)

    return p_value.to(dtype_origin)


def detect_watermark(
        z: torch.Tensor,  # 待检测的潜在向量 [B, D]
        carrier: torch.Tensor,  # 密钥向量（需单位化）[1, D]
        angle: float,  # 超锥体角度θ（弧度）
        return_confidence: bool = False,  # 是否返回置信度指标
        eps: float = 1e-7  # 数值稳定性系数
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    零比特水印检测函数
    返回：
        if return_confidence=False:
            mask [B] - 布尔张量，True表示检测到水印
        else:
            (mask [B], R [B], log10_pvalue [B]) - 附加置信度信息
    """
    # ------------------------------
    # 1. 输入验证与预处理
    # ------------------------------
    assert z.dim() == 2, f"潜在向量 z 必须是二维张量 [B, D]，当前维度 {z.dim()}"
    assert carrier.dim() == 2 and carrier.shape[0] == 1, \
        f"密钥 carrier 必须是二维张量 [1, D]，当前形状 {carrier.shape}"
    assert z.size(1) == carrier.size(1), \
        f"潜在向量维度不匹配: z[{z.size(1)}] vs carrier[{carrier.size(1)}]"
    # 确保carrier是单位向量（L2范数归一化）
    carrier = F.normalize(carrier, p=2, dim=1)  # [1, D]
    cos_theta = torch.tensor(np.cos(angle), device=z.device)  # 标量
    # ------------------------------
    # 2. 核心计算
    # ------------------------------
    # 计算点积和范数（保持维度以便广播）
    dot_product = torch.matmul(z, carrier.T)  # [B, 1]
    norm_z = torch.norm(z, p=2, dim=1, keepdim=True)  # [B, 1]
    # 计算接受函数 R = (a·z)^2 - ||z||^2 * cos^2θ （论文公式3）
    R = (dot_product ** 2) - (norm_z ** 2) * (cos_theta ** 2)  # [B, 1]
    R = R.squeeze(1)  # [B]
    # 检测条件：R > 0
    mask = R > 0
    # ------------------------------
    # 3. 置信度计算（可选）
    # ------------------------------
    if return_confidence:
        # 计算余弦相似度（添加eps防止除零）
        cosine_sim = dot_product / (norm_z + eps)  # [B, 1]
        cosine_sim = cosine_sim.squeeze(1).clamp(-1.0, 1.0)  # [B]
        # 计算p值（基于Beta分布）
        d = z.size(1)  # 潜在空间维度
        p_value = _cosine_similarity_pvalue(cosine_sim, d)
        log10_pvalue = torch.log10(p_value + eps)  # 避免log(0)
        return mask, R, log10_pvalue
    else:
        return mask

def detect_watermark_in_mu(
    mu: torch.Tensor,          # 待检测的均值向量 [B, D]
    carrier: torch.Tensor,    # 密钥向量（需单位化）[1, D]
    angle: float,             # 超锥体角度θ（弧度）
    return_confidence: bool = False,
    eps: float = 1e-7
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    直接在均值向量 μ 中检测水印（去除了噪声影响）
    """
    # 输入验证
    assert mu.dim() == 2, f"μ 必须是二维张量 [B, D]，当前维度 {mu.dim()}"
    assert carrier.dim() == 2 and carrier.shape[0] == 1, "密钥格式错误"
    assert mu.size(1) == carrier.size(1), "维度不匹配"

    # 确保密钥单位化
    carrier = F.normalize(carrier, p=2, dim=1)
    cos_theta = torch.tensor(np.cos(angle), device=mu.device)

    # 计算接受函数 R = (μ·a)^2 - ||μ||^2 cos^2θ
    dot_product = mu @ carrier.T                     # [B, 1]
    norm_mu = torch.norm(mu, dim=1, keepdim=True)    # [B, 1]
    R = (dot_product ** 2) - (norm_mu ** 2) * (cos_theta ** 2)
    mask = R.squeeze(1) > 0                          # [B]

    if return_confidence:
        # 计算置信度（μ 无噪声，p值更可靠）
        cosine_sim = dot_product / (norm_mu + eps)   # [B, 1]
        cosine_sim = cosine_sim.squeeze(1).clamp(-1.0, 1.0)
        d = mu.size(1)
        p_value = _cosine_similarity_pvalue(cosine_sim, d)
        log10_p = torch.log10(p_value + eps)
        return mask, R.squeeze(1), log10_p
    else:
        return mask