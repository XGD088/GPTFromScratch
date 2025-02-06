import time

import tiktoken
import torch

from fine_tuning.instruction.InitModel import model
from fine_tuning.instruction.InstructionDataLoader import device, train_loader, val_loader, tokenizer
from fine_tuning.instruction.InstructionDataSet import format_input_data, val_data
from pre_training.TrainModel import train_model_simple

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input_data(val_data[0]),tokenizer=tiktoken.get_encoding("gpt2")
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

