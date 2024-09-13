# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Generate synthetic data with clusters
X, y = make_blobs(n_samples=500,
                  centers=4,
                  n_features=10,
                  cluster_std=2.5,
                  random_state=42)

# Perform PCA to reduce data to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a scatter plot of the PCA-transformed data
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0],
                      X_pca[:, 1],
                      c=y,
                      cmap='viridis',
                      s=70)

# Access the current Axes instance
ax = plt.gca()

# Turn off the left and top axis spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add labels and title
plt.xlabel('PCA1', fontsize=16)
plt.ylabel('PCA2', fontsize=16)

# Create a legend
legend1 = plt.legend(*scatter.legend_elements(),
                     title="Clusters",
                     loc="upper right")
plt.gca().add_artist(legend1)

# Show the plot
plt.savefig('pca.png')
