import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dialogue representation matrix
df = pd.read_csv("dialogue_representation_matrix.csv")

# Drop rows with NaN values in 'semantic' or 'topics' columns
df = df.dropna(subset=['semantic', 'topics'])

# Combine the relevant text data into one column for vectorization
df['feature_text'] = df['semantic'] + ' ' + df['topics']

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get BERT embeddings
def get_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        # Get embeddings from BERT (take the last hidden state)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            # Use the mean of the hidden states to represent the text (CLS token could be used as well)
            sentence_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
        embeddings.append(sentence_embedding)
    return embeddings

# Get BERT embeddings for each dialogue's feature text
embeddings = get_bert_embeddings(df['feature_text'].tolist())

# Perform Agglomerative Clustering on the BERT embeddings
n_clusters = 5  # Set the number of topics (adjustable)
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')

# Fit the Agglomerative Clustering model
df['topic'] = agg_clustering.fit_predict(embeddings)

# Print out the clustering results
print("Topic modeling complete. Results:")
# print(df[['speaker', 'utterance', 'topic']])

# Print utterances in each cluster separately
for cluster in range(n_clusters):
    print(f"\nCluster {cluster}:")
    cluster_utterances = df[df['topic'] == cluster]['utterance']
    for utterance in cluster_utterances:
        print(utterance)

# Optionally, save the clustered result to a new CSV file
df.to_csv("dialogue_with_agglomerative_topics.csv", index=False)

# # Visualize the topics in 2D using PCA (for dimensionality reduction)
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(embeddings)

# plt.figure(figsize=(10, 8))
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['topic'], cmap='viridis')
# plt.title("Topic Modeling of Dialogue Segments (Using Agglomerative Clustering with BERT Embeddings)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.colorbar(label='Topic')
# plt.show()
