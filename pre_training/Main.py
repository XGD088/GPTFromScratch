import tiktoken

from pre_training.TrainModel import train_model_simple
from pre_training.TrainModelFromLoadWeight import countinueTrainFormLoadWeight
from pre_training.CalculateLossWithDataSet import device
import torch

from GPTModel import generate_text_simple, GPTModel
from pre_training.CalculateLoss import text_to_token_ids, token_ids_to_text, GPT_CONFIG_124M
from pre_training.CalculateLossWithDataSet import calc_loss_batch, calc_loss_loader, train_loader, val_loader, device

if __name__ == "__main__":
    countinueTrainFormLoadWeight(device, 1, 5, 1)



def test_gpt_model():
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  # A
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tiktoken.get_encoding("gpt2")
    )

    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
        "model_and_optimizer.pth"
    )