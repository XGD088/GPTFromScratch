from functools import partial
from torch.utils.data import DataLoader
import tiktoken
import torch

from fine_tuning.instruction.InstructionDataSet import custom_collate_fn, InstructionDataSet, train_data, val_data, \
    test_data

NUM_WORKERS = 0
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

print("Device:", device)

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

train_dataset = InstructionDataSet(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=NUM_WORKERS
)

val_dataset = InstructionDataSet(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS
)
test_dataset = InstructionDataSet(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS
)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))
print("Train loader:")
for i, (inputs, targets) in enumerate(train_loader):
    if i < 2:
        print(i, inputs.shape, targets.shape)
        print("Inputs:", str(inputs))
        print("Targets:", str(targets))
