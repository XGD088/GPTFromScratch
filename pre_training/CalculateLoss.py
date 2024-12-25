import tiktoken
import torch

from GPTConfig import GPT2_SMALL_CONFIG
from GPTModel import GPTModel
from text_processer.BPETokenizer import create_dataloader_v1


def text_to_token_ids(text, tokenizer):
    encoded_text = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded_text).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat_list = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat_list)


#减小了context_length的长度
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}


torch.manual_seed(123)
model = GPTModel(GPT2_SMALL_CONFIG)
model.eval()

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")


inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                       [40, 1107, 588]]) # "I really like"]

targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                        [588, 428, 11311]]) # " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas.shape)
token_ids = torch.argmax(probas, dim=-1)
generated_text_batch1 = token_ids_to_text(token_ids[0].flatten(), tokenizer)
input_text_batch1 = token_ids_to_text(inputs[0].flatten(), tokenizer)
print(f'Targets batch 1:{input_text_batch1}')
print(f'Outputs batch 1:{generated_text_batch1}')

input_text_idx = 0
target_probas_1 = probas[input_text_idx, [0, 1, 2], targets[input_text_idx]]

input_text_idx = 1
target_probas_2 = probas[input_text_idx, [0, 1, 2], targets[input_text_idx]]

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))

avg_log_probas = torch.mean(log_probas, dim=-1)

loss = avg_log_probas * -1


logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
loss_2 = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

print(loss)
print(loss_2)
