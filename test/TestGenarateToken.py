import tiktoken
import torch

from GPTConfig import GPT2_SMALL_CONFIG
from GPTModel import generate_text_simple, GPTModel

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model = GPTModel(GPT2_SMALL_CONFIG)

model.eval() # disable dropout

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT2_SMALL_CONFIG["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
print("Output[0]",out.squeeze(0))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)