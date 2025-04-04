import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("model moved to gpu")

# Read your transcript from file
with open(r"C:\Users\shambhawi\Source\Repos\cluster_study1\Case Study-1.txt", "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file.readlines() if line.strip()]

# Filter to only dialogue lines
dialogues = [line for line in lines if ':' in line]
print(dialogues)

# Define prompt template
def create_prompt(utterance):
    return (
        f"Analyze the following sentence and extract the following:\n"
        f"1. Semantic meaning (e.g., complaint, justification, defense, evidence)\n"
        f"2. Tone (e.g., frustration, denial, neutral)\n"
        f"3. Topic categories: what is the topic of this statement\n\n"
        f"Sentence: \"{utterance}\"\n"
        f"Return in JSON format with keys: semantic, tone, topics."
    )

# Store results
rows = []

for line in dialogues:
    speaker, utterance = line.split(":", 1)
    prompt = create_prompt(utterance.strip())

    # Tokenize and get response from model
    inputs = tokenizer(text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Debug print to see the raw response
    print(f"Raw response: {response}")

    # Attempt to parse the response
    try:
        # Extract the JSON part from the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_response = response[json_start:json_end]
            result = json.loads(json_response)  # Use json.loads to parse the response
        else:
            raise ValueError("JSON part not found in the response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse response: {response}")
        result = {"semantic": "unknown", "tone": "unknown", "topics": []}

    print(f"Parsed result: {result}")
    rows.append({
        "speaker": speaker.strip(),
        "utterance": utterance.strip(),
        "semantic": result.get("semantic", "unknown"),
        "tone": result.get("tone", "unknown"),
        "topics": ", ".join(result.get("topics", [])),  # Join list of topics as a string
    })

# Convert to DataFrame
df = pd.DataFrame(rows)
print("representation matrix:")

# Save to CSV or use for clustering
df.to_csv("dialogue_representation_matrix.csv", index=False)
print(df.head())
