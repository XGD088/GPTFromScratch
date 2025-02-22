import tiktoken
import torch

from GPTModel import GPTModel
from pre_training.CalculateLoss import GPT_CONFIG_124M
from pre_training.CalculateLossWithDataSet import train_loader, val_loader, device
from pre_training.TrainModel import train_model_simple
from pre_training.TrainModelFromLoadWeight import countinueTrainFormLoadWeight, model_path, model_and_optimizer_path


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

    torch.save(model.state_dict(), model_path)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
        model_and_optimizer_path
    )


if __name__ == "__main__":
    test_gpt_model()
    countinueTrainFormLoadWeight(device, 1, 5, 1)




