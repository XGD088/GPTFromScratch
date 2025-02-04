import torch

from fine_tuning.instruction.DownloadDataSet import download_and_load_file


class InstructionDataSet:
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instructions_plus_input = format_input_data_in_alpaca_style(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            encoded_text = tokenizer.encode(instructions_plus_input + response_text)
            self.encoded_texts.append(encoded_text)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def format_input_data_in_alpaca_style(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text


def partition_data_set(data, train_ratio, val_ratio):
    num_train = int(len(data) * train_ratio)
    num_val = int(len(data) * val_ratio)

    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]

    return train_data, val_data, test_data


def custom_collate_draft_fn(batch, ignore_index=-100, allowed_max_length=None,
                            pad_token_id=50256, device="cpu"):
    # max_len plus one for conveniently process the target tensor
    max_length = max([len(entry) + 1 for entry in batch])
    inputs_tensor_list, targets_tensor_list = [], []
    for entry in batch:
        padded_entry = entry + [pad_token_id] * (max_length - len(entry))
        inputs_tensor_item = torch.tensor(padded_entry[:-1])
        targets_tensor_item = torch.tensor(padded_entry[1:])

        mask = targets_tensor_item == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets_tensor_item[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs_tensor_item = inputs_tensor_item[:allowed_max_length]
        targets_tensor_item = targets_tensor_item[:allowed_max_length]


        inputs_tensor_list.append(inputs_tensor_item)
        targets_tensor_list.append(targets_tensor_item)

    # stack tensors and move to device
    return torch.stack(inputs_tensor_list).to(device), torch.stack(targets_tensor_list).to(device)


def custom_collate_draft_2(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json")

data = download_and_load_file(file_path, url)
model_input = format_input_data_in_alpaca_style(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)
train_data, val_data, test_data = partition_data_set(data, 0.85, 0.05)
print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of test samples: {len(test_data)}")

inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
    inputs_1,
    inputs_2,
    inputs_3
)
draft_1 = custom_collate_draft_fn(batch)

print("target tokens:\n",draft_1[1])
