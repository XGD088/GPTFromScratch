import pandas as pd
import urllib.request
import zipfile
import os
from pathlib import Path


def create_balance_dataset(df):
        nums_spam = df[df["Label"] == "spam"].shape[0]
        hum_subset = df[df["Label"] == "ham"].sample(nums_spam, random_state=123)
        balance_df = pd.concat([hum_subset, df[df["Label"] == "spam"]])
        return balance_df

def random_split_data(df, train_frac, validation_frac):

    # shuffle the data
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(train_frac * len(df))
    val_end = int(validation_frac * len(df)) + train_end

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df




url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction."
              )
        return

    # Download the zip file
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as file:
            file.write(response.read())

    # Unzips the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
