
from torchvision import transforms as tvt

import torch
from PolarTransformer import PolarTransformer


def test_polar_transform():
    # 使用低分辨率测试（减少计算量）
    h, w = 32, 32
    transformer = PolarTransformer(input_shape=(h, w))

    # 生成测试图像（平滑渐变减少高频噪声）
    x = torch.linspace(0, 1, w).reshape(1, 1, w).expand(1, 3, h, w)
    y = torch.linspace(0, 1, h).reshape(1, 1, h).expand(1, 3, h, w)
    test_img = (x + y) / 2  # [0,1]渐变

    # 转换测试
    polar = transformer.to_polar(test_img)
    reconstructed = transformer.to_cartesian(polar)

    # 计算相对误差（允许1%误差）
    abs_error = torch.abs(test_img - reconstructed).mean()
    relative_error = abs_error / test_img.mean()
    assert relative_error < 0.01, f"Relative error {relative_error:.4f} exceeds 1%"

    # 旋转不变性测试（30度旋转）
    rotated = tvt.functional.rotate(test_img, 30)
    polar_rotated = transformer.to_polar(rotated)

    # 应表现为环形位移
    shift = int(30 / 360 * polar.size(3))
    shifted_polar = torch.roll(polar, shifts=shift, dims=3)
    rotation_error = (shifted_polar - polar_rotated).abs().mean()
    assert rotation_error < 0.05, "Rotation not equivalent to polar shift"


if __name__=="__main__":
    test_polar_transform()