import tiktoken

from GPTModel import GPTModel, GPT_CONFIG_124M
import torch


batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

tokenizer = tiktoken.get_encoding("gpt2")
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)