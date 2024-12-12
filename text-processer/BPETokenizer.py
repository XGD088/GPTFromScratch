import re
import tiktoken
from torch.utils.data import DataLoader

from GPTDataSet import GPTDatasetV1
def create_dataloader_v1(txt, batch_size=4,
        max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) #C
    return dataloader



with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    for i in range(0, 3):
        first_batch = next(data_iter)
        print(first_batch)




