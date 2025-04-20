# from huggingface_hub import snapshot_download
# snapshot_download(
#   repo_id="svjack/pokemon-blip-captions-en-zh",
#   local_dir="./svjack/pokemon-blip-captions-en-zh",
#   proxies={"https": "http://localhost:7890"},
#   max_workers=8
# )
import torchvision
print(torchvision.__version__)  # 输出 torchvision 版本
print(torchvision.models.VGG16_Weights.DEFAULT)  # 输出默认权重版本（>=0.13+）
import os

# 设置全局缓存目录（对所有torch下载生效）
os.environ['TORCH_HOME'] = '../vgg/'
print("Load VGG")
loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
loss_fn_alex.requires_grad_(False)
# 或临时修改下载路径
from torchvision.models import vgg16
vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)