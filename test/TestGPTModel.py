import tiktoken

from GPTModel import GPTModel
from GPTConfig import GPT2_SMALL_CONFIG, GPT2_LARGE_CONFIG
import torch

def count_att_ff_parameters(trf_blocks):
    total_att_params = 0
    total_feedforward_params = 0
    for block in trf_blocks:
        if hasattr(block, 'att'):  # 检查是否有 att 属性
            att_module = block.att
            att_params = sum(p.numel() for p in att_module.parameters())
            total_att_params += att_params
        if hasattr(block, 'ff'):  # 检查是否有 att 属性
            ff_module = block.ff
            att_params = sum(p.numel() for p in ff_module.parameters())
            total_feedforward_params += att_params

    return total_att_params, total_feedforward_params


batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

tokenizer = tiktoken.get_encoding("gpt2")
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)


torch.manual_seed(123)
model = GPTModel(GPT2_SMALL_CONFIG)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
# total parameters number is 163M, because GPT2 original size is using the "weight tying" way
print(f"Total number of parameters: {total_params:,}")


multi_heads_att_params, ff_params = count_att_ff_parameters(model.trf_blocks)
print(f"multi-head attention module params number of parameters: {multi_heads_att_params:,}")
print(f"feed forward module params number of parameters: {ff_params:,}")


large_model = GPTModel(GPT2_LARGE_CONFIG)

large_total_params = sum(p.numel() for p in large_model.parameters())
# 613M
print(f"GPT2 Large Total number of parameters: {large_total_params:,}")

