import json
import torch
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

_llm_cache = None
def get_llm():
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = load_llm()
    return _llm_cache


def generate_response(prompt: str, max_length: int = 512) -> str:
    tokenizer, llm_model, llm_device = get_llm()
    inputs = tokenizer(prompt, return_tensors="pt").to(llm_device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def classify_event_type(utterance: str) -> str:
    prompt = (
        "Extract a concise event type label that best describes the following HR transcript utterance. "
        "Return only the label.\n\n"
        f"Utterance: {utterance.strip()}\nLabel:"
    )
    raw = generate_response(prompt, max_length=50)
    label = raw.split('Label:')[-1].strip().splitlines()[0]
    return label

def chunk_by_event(text: str) -> list:
    chunks = []
    for line in text.splitlines():
        if ':' in line:
            utterance = line.split(':', 1)[1].strip()
            if utterance:
                evt = classify_event_type(utterance)
                chunks.append({'text': utterance, 'event_type': evt})
    return chunks

def chunk_by_timeline(text: str) -> list:
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

# load sentence‐transformer once
embed_model = SentenceTransformer('all-mpnet-base-v2')

def embed_justifications(user_justifications: dict) -> dict:
    return { law: embed_model.encode(text) for law, text in user_justifications.items() }

def embed_goals(goal_prompts: dict) -> dict:
    return { goal: embed_model.encode(prompt) for goal, prompt in goal_prompts.items() }

def match_laws_to_chunks(chunks: list, just_vecs: dict, threshold: float = 0.3) -> None:
    chunk_vecs = np.vstack([embed_model.encode(c['text']) for c in chunks])
    law_names = list(just_vecs.keys())
    law_vecs = np.vstack([just_vecs[law] for law in law_names])
    sim = cosine_similarity(chunk_vecs, law_vecs)
    for i, chunk in enumerate(chunks):
        chunk['laws'] = [law_names[j] for j, score in enumerate(sim[i]) if score >= threshold]

def build_final_vectors(chunks: list, just_vecs: dict, goal_vecs: dict,
                        alpha: float = 0.5, gamma: float = 0.2) -> np.ndarray:
    vecs = []
    for chunk in chunks:
        c = embed_model.encode(chunk['text'])
        for law in chunk.get('laws', []):
            c += alpha * just_vecs[law]
        for g in goal_vecs.values():
            c += gamma * g
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        vecs.append(c)
    return np.vstack(vecs)

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return clustering.fit_predict(embeddings)

def name_cluster(texts: list) -> str:
    prompt = (
        "Here are several HR transcript excerpts. Provide a concise theme name.\n"
        + "\n".join(texts[:5])
        + "\nTheme name:"
    )
    raw = generate_response(prompt, max_length=30)
    return raw.split('Theme name:')[-1].strip().splitlines()[0]

def cluster_pipeline(transcript: str,
                     user_justifications: dict,
                     grouping: str,
                     evaluation_goals: list,
                     n_clusters: int = 5) -> list:
    # 1. chunk
    if grouping == 'event':
        chunks = chunk_by_event(transcript)
    else:
        chunks = chunk_by_timeline(transcript)
    # 2. embed just & goals
    just_vecs = embed_justifications(user_justifications)
    goal_prompts = {}
    for ev in evaluation_goals:
        if ev == 'behavior':
            goal_prompts['Behavior patterns'] = 'behavior patterns in workplace dialogue'
        elif ev == 'power':
            goal_prompts['Power dynamics'] = 'power imbalance in managerial communication'
        else:
            goal_prompts[ev] = ev
    goal_vecs = embed_goals(goal_prompts)
    # 3. match laws
    match_laws_to_chunks(chunks, just_vecs)
    # 4. fuse
    embeddings = build_final_vectors(chunks, just_vecs, goal_vecs)
    # 5. cluster
    labels = cluster_embeddings(embeddings, n_clusters)
    # 6. group & name
    clusters = defaultdict(list)
    for chunk, label in zip(chunks, labels):
        clusters[label].append(chunk['text'])
    result = []
    for label, texts in clusters.items():
        result.append({
            'id': label,
            'title': name_cluster(texts),
            'texts': texts
        })
    return result
