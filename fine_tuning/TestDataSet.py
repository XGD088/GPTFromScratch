import pandas as pd

from fine_tuning.PrepareDataSet import data_file_path
from fine_tuning.PrepareDataSet import random_split_data, create_balance_dataset

df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)
print(df["Label"].value_counts())
balanced_df = create_balance_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

print(balanced_df["Label"].value_counts())
print(balanced_df.head())

train_df, val_df, test_df = random_split_data(balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
val_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))