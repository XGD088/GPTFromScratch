import json

import psutil
import urllib.request
import urllib.error

from tqdm import tqdm

from fine_tuning.instruction.InstructionDataSet import format_input_data


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {"model": model,
            "messages": [{"role": "user","content": prompt}],
            "stream": False,
            "options": {"seed":123, "temperature": 0, "num_ctx": 2048}}
    payload = json.dumps(data).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                response_body = json.loads(response.read().decode("utf-8"))
                return response_body["message"]["content"]
            else:
                raise RuntimeError(f"request fail, response status：{response.status}")

    except urllib.error.URLError as e:
        raise RuntimeError(f"connect fail: {e.reason}")




ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError(
        "Ollama not running. Launch ollama before proceeding."
    )
print("Ollama running:", check_if_running("ollama"))

model = "llama3"
# result = query_model("What do Llamas eat?", model)
# print(result)



test_data_with_response = json.load(open("instruction-data-with-response.json", "r"))
for entry in test_data_with_response[:3]:
    prompt = (
    f"Given the input `{format_input_data(entry)}` "
    f"and correct output `{entry['output']}`, "
    f"score the model response `{entry['model_response']}`"
    f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring Entries"):
        prompt = (
            f"Given the input `{format_input_data(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Please provide a score for the model response, and respond with the integer score only."
        )
        score = query_model(prompt)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Invalid score: {score}")
            continue
    return scores


scores = generate_model_scores(test_data_with_response, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data_with_response)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")