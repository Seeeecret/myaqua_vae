import torch
import torch.nn as nn
import numpy as np


class TrainFreeMapper(nn.Module):  # 需继承nn.Module
    def __init__(self, input_size=48, output_size=64, std=1.0):
        """
        参数说明同原MapperNet：
        - input_size: 输入二进制位数
        - output_size: 输出维度
        - std: 输出幅度控制
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.std = std
        # 生成固定正交基（替代可训练嵌入层）

        basis = self._generate_orthogonal_basis()
        self.register_buffer('basis', basis)  # 关键注册操作

    # 哈达玛矩阵生成函数（需自行实现）
    # def hadamard_matrix(n):
    #     H = [[1]]
    #     while len(H) < n:
    #         H = np.kron(H, [[1, 1], [1, -1]])
    #     return H[:n, :n]

    import numpy as np
    import torch

    def hadamard_matrix(self, n, method='kronecker'):
        """
        生成n阶哈达玛矩阵的两种方法
        参数：
            n : 目标矩阵阶数（建议为2的幂）
            method : 'kronecker'（克罗内克积法）或 'silvester'（西尔维斯特法）
        返回：
            H : (n,n) numpy数组
        """
        assert method in ['kronecker', 'silvester'], "无效方法选择"

        # 基础哈达玛矩阵
        H2 = np.array([[1, 1], [1, -1]])

        if method == 'kronecker':
            # 递归构造法（适用于任意2^k阶）
            k = int(np.ceil(np.log2(n)))
            H = H2.copy()
            for _ in range(k - 1):
                H = np.kron(H, H2)
            return H[:n, :n]  # 截断到目标尺寸

        elif method == 'silvester':
            # 西尔维斯特构造法（仅生成2^k阶）
            m = int(2 ** np.ceil(np.log2(n)))
            H = H2
            while H.shape[0] < m:
                H = np.block([[H, H],
                              [H, -H]])
            return H[:n, :n]

    def _generate_orthogonal_basis(self):
        """生成确定性正交基的两种方法"""
        # 方法1：哈达玛矩阵（要求output_size是2的幂）
        if (self.output_size & (self.output_size - 1)) == 0:  # 检查是否为2的幂
            H = torch.tensor(self.hadamard_matrix(self.output_size, method='silvester')).float()
            return H[:self.input_size]  # 取前input_size行
        # 方法2：QR分解生成随机正交基
        random_matrix = torch.randn(self.input_size, self.output_size)
        q, _ = torch.linalg.qr(random_matrix, mode='reduced')
        return q * np.sqrt(self.output_size)  # 调整幅度

    def __call__(self, x):
        """
        输入x: [batch_size, input_size] ∈ {0,1}
        输出: [batch_size, output_size]
        """
        # 二进制门控（与原始实现相同）
        x = x.float()
        encoded = self.basis.to(x.device)  # [input_size, output_size]
        encoded = encoded[None, :, :] * x[:, :, None]  # 广播乘法
        # 聚合与归一化
        agg = encoded.sum(dim=1) / np.sqrt(self.input_size)
        # 幅度控制与偏置
        output = agg * self.std + 1.0
        return output
