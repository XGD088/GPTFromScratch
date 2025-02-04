import json
import os
import urllib


def download_and_load_file(file_path, url):
    # Download and save the file if it doesn't exist
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    else:
        print(f"{file_path} already exists. Skipping download.")

    # Load the file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


