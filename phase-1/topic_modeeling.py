import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Path to the case study
case_study_path = r"C:\Users\shambhawi\Source\Repos\cluster_study1\Case Study-1.txt"

# Load and preprocess the case study
with open(case_study_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Keep only dialogue lines (lines with ':')
dialogues = [line for line in lines if ':' in line]

# Optional: Chunk dialogues into larger blocks for better topic granularity
# You can adjust the chunk size depending on how broad/narrow you want topics to be
chunk_size = 4  # number of utterances per chunk
chunks = [' '.join(dialogues[i:i+chunk_size]) for i in range(0, len(dialogues), chunk_size)]

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get BERT embeddings for each chunk
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of the last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    return embeddings

embeddings = get_bert_embeddings(chunks)

# Perform Agglomerative Clustering
n_clusters = 5  # Set number of topics (tune this based on your case)
clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
cluster_labels = clustering_model.fit_predict(embeddings)

# Save results
df = pd.DataFrame({
    "chunk": chunks,
    "topic": cluster_labels
})

df.to_csv("case_study_topic_modeling.csv", index=False)
print("Topic modeling complete. Results:")
# print(df)

# Print utterances in each cluster separately
for cluster in range(n_clusters):
    print(f"\nCluster {cluster}:")
    cluster_chunks = df[df['topic'] == cluster]['chunk']
    for chunk in cluster_chunks:
        print(chunk)

# Optional: Visualize using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='tab10')
for i, txt in enumerate(cluster_labels):
    plt.annotate(txt, (reduced[i, 0], reduced[i, 1]))
plt.title("Case Study Topic Modeling (Agglomerative + BERT)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster/Topic ID')
plt.show()
