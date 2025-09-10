import torch
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)

device = 'cuda'
torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet.to(device)
birefnet.eval()