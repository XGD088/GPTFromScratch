import torch

from GPTModel import TransformerBlock
from GPTConfig import GPT2_SMALL_CONFIG

torch.manual_seed(123)
x = torch.rand(2, 4, 768) #A
block = TransformerBlock(GPT2_SMALL_CONFIG)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)