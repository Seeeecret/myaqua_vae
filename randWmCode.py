import torch
from accelerate.utils import set_seed

set_seed(2024)
msg_bits = 48
wm = torch.randint(0, 2, (1, msg_bits))
print(wm)

# 将wm转换为字符串
wm_str = ''.join([str(i) for i in wm.tolist()[0]])
print(wm_str)