import json

import torch
from tqdm import tqdm

from fine_tuning.instruction.InitModel import BASE_CONFIG, model
from fine_tuning.instruction.InstructionDataLoader import tokenizer, train_data, device, test_data
from fine_tuning.instruction.InstructionDataSet import format_input_data, custom_collate_fn
from pre_training.CalculateLoss import text_to_token_ids, token_ids_to_text
from pre_training.LoadWeightFromOpenAI import generate

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
inputs_tensors, target_tensors = custom_collate_fn(batch)

print("inputs_tensors:\n", inputs_tensors)
print("target_tensors:\n", target_tensors)

torch.manual_seed(123)
for entry in train_data[:3]:
    instructions_plus_input = format_input_data(entry)
    generated_token_ids = generate(model=model,
                                   idx=text_to_token_ids(instructions_plus_input, tokenizer),
                                   max_new_tokens=35,
                                   context_size=BASE_CONFIG["context_length"],
                                   eos_id=50256)
    generated_text = token_ids_to_text(generated_token_ids, tokenizer)
    response_text = (generated_text[len(instructions_plus_input):]
                     .replace("### Response:", "")
                     .strip())
    print("Instruction:\n", instructions_plus_input)
    print("Correct response:\n", entry["output"])
    print("Model Generated response:\n", response_text)

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input_data(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)
