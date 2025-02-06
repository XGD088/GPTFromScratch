import torch

from GPTModel import GPTModel
from fine_tuning.instruction.InstructionDataLoader import tokenizer, device, train_loader, val_loader, val_data
from fine_tuning.instruction.InstructionDataSet import format_input_data
from pre_training.CalculateLoss import token_ids_to_text, text_to_token_ids
from pre_training.CalculateLossWithDataSet import calc_loss_loader
from pre_training.LoadWeightFromOpenAI import load_weights_into_gpt, generate
from pre_training.gpt_download import download_and_load_gpt2

CHOOSE_MODEL = "gpt2-medium (355M)"
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
    model_size=model_size, models_dir="../../resources/gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.load_state_dict(torch.load("gpt2-medium355M-sft.pth", map_location=device, weights_only=False))
print("model loaded")
# model.eval()

# torch.manual_seed(123)
# input_text = format_input_data(val_data[0])
# print("input_text:\n", input_text)
#
# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids(input_text, tokenizer),
#     max_new_tokens=35,
#     context_size=BASE_CONFIG["context_length"],
#     eos_id=50256,
# )
#
# # 生成的文本中包含了输入文本
# generated_text = token_ids_to_text(token_ids, tokenizer)
# # 截取response部分
# response_text = generated_text[len(input_text):]
# print("Response text:\n", response_text)
#
# model.to(device)
# torch.manual_seed(123)
# with torch.no_grad():
#     train_loss = calc_loss_loader(
#         train_loader, model, device, num_batches=5
#     )
# val_loss = calc_loss_loader(
#     val_loader, model, device, num_batches=5
# )
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)
