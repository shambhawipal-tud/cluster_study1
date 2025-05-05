import json
import torch
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# 1. Configure local LLM
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE = 0.3

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, llm_model, llm_device = load_llm()

# Generate raw LLM response
def generate_response(prompt: str, max_length: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 2. Load the embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# 3. Utterance extraction & dynamic event labeling
def classify_event_type(utterance: str) -> str:
    """Use LLM to extract an open-ended event type label for an utterance."""
    prompt = (
        "Extract a concise event type label that best describes the following HR transcript utterance. "
        "Return only the label.\n\n"
        "Utterance: " + utterance.strip() + "\nLabel:"
    )
    raw = generate_response(prompt, max_length=50)
    # Grab the first non-empty line after the Label prompt
    label = raw.split('Label:')[-1].strip().splitlines()[0]
    return label


def chunk_by_event(text: str) -> list:
    """Splits transcript into utterances and labels each with a dynamic event type."""
    chunks = []
    for line in text.splitlines():
        if ':' in line:
            parts = line.split(':', 1)
            utterance = parts[1].strip()
            if utterance:
                evt = classify_event_type(utterance)
                chunks.append({'text': utterance, 'event_type': evt})
    return chunks


def chunk_by_timeline(text: str) -> list:
    """Splits transcript into before/after complaint phases, then into utterances."""
    lower = text.lower()
    idx = lower.find('complaint')
    if idx == -1:
        phase_texts = [('full', text)]
    else:
        phase_texts = [('before', text[:idx]), ('after', text[idx:])]
    chunks = []
    for phase, segment in phase_texts:
        for line in segment.splitlines():
            if ':' in line:
                utterance = line.split(':',1)[1].strip()
                if utterance:
                    chunks.append({'text': utterance, 'time_phase': phase})
    return chunks

# 4. Embed justifications (Q1)
def embed_justifications(user_justifications: dict) -> dict:
    return { law: model.encode(txt) for law, txt in user_justifications.items() }

# 5. Embed evaluation goals (Q3)
def embed_goals(goal_prompts: dict) -> dict:
    return { goal: model.encode(prompt) for goal, prompt in goal_prompts.items() }

# 6. Match laws to chunks by cosine similarity
def match_laws_to_chunks(chunks: list, just_vecs: dict, threshold: float = 0.3) -> None:
    chunk_vecs = np.vstack([model.encode(c['text']) for c in chunks])
    law_names = list(just_vecs.keys())
    law_vecs = np.vstack([just_vecs[law] for law in law_names])
    sim = cosine_similarity(chunk_vecs, law_vecs)
    for i, chunk in enumerate(chunks):
        chunk['laws'] = [law_names[j] for j, score in enumerate(sim[i]) if score >= threshold]

# 7. Build final fused vectors
def build_final_vectors(chunks: list, just_vecs: dict, goal_vecs: dict,
                        alpha: float = 0.5, gamma: float = 0.2) -> np.ndarray:
    vecs = []
    for chunk in chunks:
        c = model.encode(chunk['text'])
        for law in chunk.get('laws', []):
            c += alpha * just_vecs[law]
        for g in goal_vecs.values():
            c += gamma * g
        vecs.append(c / np.linalg.norm(c))
    return np.vstack(vecs)

# 8. Clustering pipeline using Agglomerative Clustering
def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return clustering.fit_predict(embeddings)

# 9. Name clusters via LLM
def name_cluster(texts: list) -> str:
    prompt = (
        "Here are several HR transcript excerpts. Provide a concise theme name.\n" +
        "\n".join(texts[:5]) + "\nTheme name:"  
    )
    raw = generate_response(prompt, max_length=30)
    return raw.split('Theme name:')[-1].strip().splitlines()[0]

# --- Example usage ---
if __name__ == '__main__':
    # Load transcript
    with open(r"C:\Users\shambhawi\Source\Repos\cluster_study1\Case Study-1.txt", 'r', encoding='utf-8') as f:
        transcript = f.read()

    # Q2: Choose grouping
    grouping = 'event'
    chunks = chunk_by_event(transcript) if grouping=='event' else chunk_by_timeline(transcript)

    # Q1: Justifications
    user_justifications = {
        'Harassment': 'The late-night messages felt inappropriate and targeted Sophia.',
        'Retaliation': 'Workload visibly increased after the informal concerns were raised.'
    }
    just_vecs = embed_justifications(user_justifications)

    # Q3: Goals
    goal_prompts = {
        'Behavior patterns': 'behavior patterns in workplace dialogue',
        'Power dynamics': 'power imbalance in managerial communication'
    }
    goal_vecs = embed_goals(goal_prompts)

    # Process & cluster
    match_laws_to_chunks(chunks, just_vecs)
    embs = build_final_vectors(chunks, just_vecs, goal_vecs)
    labels = cluster_embeddings(embs, n_clusters=5)

    # Group and print
    clusters = defaultdict(list)
    for c, lab in zip(chunks, labels):
        clusters[lab].append(c['text'])

    for lab, texts in clusters.items():
        title = name_cluster(texts)
        print(f"\n=== Cluster {lab}: {title} ===")
        for utt in texts:
            print(f"- {utt}")

