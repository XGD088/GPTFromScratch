import torch

from GPTModel import GPTModel, generate_text_simple
from fine_tuning.SpamDataLoader import train_loader, val_loader, test_loader
from pre_training.CalculateLoss import text_to_token_ids, token_ids_to_text, tokenizer
from pre_training.CalculateLossWithDataSet import device
from pre_training.LoadWeightFromOpenAI import load_weights_into_gpt
from pre_training.gpt_download import download_and_load_gpt2

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()


num_classes = 2

model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

print("Last output token:", outputs[:, -1, :])

logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        else:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
        total_loss += loss.item()

    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                # [batch, seq_len, num_classes],
                # seq_len是序列长度，取最后一个是因为这里是考虑整个序列的结果，注意力与前馈神经网络在这里会累积整个序列的信息
                # 这里的num_classes是最后一层的输出，也就是分类结果的概率
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            correct += (predicted_labels == target_batch).sum().item()
            total += target_batch.numel()
        else:
            break
    return correct / total


with torch.no_grad():
    accuracy_train = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    accuracy_val = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    accuracy_test = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    print(f"Accuracy on training set: {accuracy_train * 100:.2f}")
    print(f"Accuracy on validation set: {accuracy_val * 100:.2f}")
    print(f"Accuracy on test set: {accuracy_test * 100:.2f}")

with torch.no_grad():
    loss_train = calc_loss_loader(train_loader, model, device, num_batches=5)
    loss_val = calc_loss_loader(val_loader, model, device, num_batches=5)
    loss_test = calc_loss_loader(test_loader, model, device, num_batches=5)
    print(f"loss on training set: {loss_train:.2f}")
    print(f"loss on validation set: {loss_val:.2f}")
    print(f"loss on test set: {loss_test:.2f}")