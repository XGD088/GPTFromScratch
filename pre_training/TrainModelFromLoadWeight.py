import os

import tiktoken
import torch

from GPTModel import GPTModel
from pre_training.CalculateLoss import GPT_CONFIG_124M
from pre_training.CalculateLossWithDataSet import train_loader, val_loader, device
from pre_training.TrainModel import train_model_simple
from utils.os_util import get_current_dir

model_path = os.path.join(get_current_dir(), '../resources/pre_trained_model.pth')
model_and_optimizer_path = os.path.join(get_current_dir(), '../resources/model_and_optimizer.pth')


def countinueTrainFormLoadWeight(device, num_epochs, eval_freq, eval_iter):

    checkpoint = torch.load(model_and_optimizer_path, map_location=device)
    model = GPTModel(GPT_CONFIG_124M)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context="Every effort moves you", tokenizer=tiktoken.get_encoding("gpt2")
    )



