import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Load the dialogue representation matrix
df = pd.read_csv("dialogue_representation_matrix.csv")

# Drop rows with NaN values in 'semantic', 'tone', or 'topics' columns
df = df.dropna(subset=['semantic', 'tone', 'topics'])

# Combine all relevant text data into one column for vectorization
df['feature_text'] = df['semantic'] + ' ' + df['topics']

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the feature text to obtain the TF-IDF matrix
X = vectorizer.fit_transform(df['feature_text'])

# 2. Apply Agglomerative Clustering

# Define the Agglomerative Clustering model
n_clusters = 5  # Set the number of clusters (this can be adjusted or based on a criterion)
clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')

# Fit the model on the TF-IDF matrix
df['cluster'] = clustering_model.fit_predict(X.toarray())

# 3. Add the cluster labels to the dataframe
print("Clustering complete. Results:")
print(df[['speaker', 'utterance', 'cluster']])

# Optionally, save the clustered result to a new CSV file
df.to_csv("dialogue_with_clusters.csv", index=False)

# 4. To visualize the clustering, you can use something like a scatter plot if you want
# This will reduce the dimensionality for plotting purposes (e.g., using PCA or t-SNE)
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# Reduce the dimensionality to 2D for visualization
# pca = PCA(n_components=2)
# reduced_data = pca.fit_transform(X.toarray())

# Plot the clusters
# plt.figure(figsize=(10, 8))
# plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df['cluster'], cmap='viridis')
# plt.title("Agglomerative Clustering of Dialogue Segments")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.colorbar(label='Cluster')
# plt.show()
