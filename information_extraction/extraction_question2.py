import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE = 0.3

GRAPHDB_ENDPOINT = "http://127.0.0.1:7200/repositories/group_discussion/statements"
HEADERS = {
    "Content-Type": "application/sparql-update"
}

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def get_llm_response(tokenizer, model, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=256,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Trim the prompt from the response
    return response[len(prompt):].strip()

def sanitize_uri(text):
    """
    Shortens and sanitizes text to create a URI-safe slug.
    """
    short_text = text.split(".")[0][:100]
    sanitized = short_text.replace(" ", "_").replace("(", "").replace(")", "").replace('"', "").replace(",", "").replace("'", "").replace(".", "")
    return sanitized

def parse_dialogue(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    dialogues = []
    for line in lines:
        if ':' in line:
            participant, utterance = line.split(":", 1)
            dialogues.append((participant.strip(), utterance.strip()))
    return dialogues

def extract_events(dialogue_lines, tokenizer, model, device):
    prompt = (
    "From the following case study dialogue, identify all major events or themes mentioned, however, try to keep the theme as big as possible and do not list a huge set of themes.\n"
    "List them clearly in a numbered format with no additional text or explanation.\n\n"
    "Dialogue:\n" + "\n".join(dialogue_lines) + "\n\n"
    "List of events or themes in a json format:"
)


    response = get_llm_response(tokenizer, model, device, prompt)
    return response.split("\n")

def group_by_speaker(dialogues):
    grouped = {}
    for participant, utterance in dialogues:
        if participant not in grouped:
            grouped[participant] = []
        grouped[participant].append(utterance)
    return grouped

def group_by_event(dialogues, tokenizer, model, device):
    dialogue_lines = [utterance for _, utterance in dialogues]
    events = extract_events(dialogue_lines, tokenizer, model, device)
    
    grouped_by_event = {}
    for event in events:
        grouped_by_event[event] = []
        for participant, utterance in dialogues:
            if event.strip() and event in utterance:
                grouped_by_event[event].append(utterance)
    return grouped_by_event

def group_by_timeline(dialogues):
    grouped = {"before_event": [], "after_event": []}
    for idx, (participant, utterance) in enumerate(dialogues):
        if 'complaint' in utterance.lower():
            grouped['before_event'] = dialogues[:idx]
            grouped['after_event'] = dialogues[idx:]
            break
    return grouped

def process_dialogue(file_path, grouping_type, tokenizer, model, device):
    dialogues = parse_dialogue(file_path)
    
    if grouping_type == "speaker":
        return group_by_speaker(dialogues)
    elif grouping_type == "event":
        return group_by_event(dialogues, tokenizer, model, device)
    elif grouping_type == "timeline":
        return group_by_timeline(dialogues)
    else:
        return {}

if __name__ == "__main__":
    tokenizer, model, device = load_llm()

    # Example input (file path & grouping type as per user selection)
    file_path = r"C:\Users\shambhawi\Source\Repos\cluster_study1\Case Study-1.txt"
    grouping_type = "event"  # Could be "speaker", "event", or "timeline"

    grouped_data = process_dialogue(file_path, grouping_type, tokenizer, model, device)

    # Pass this to your Flask route or use directly
    print(grouped_data)
