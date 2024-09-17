import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from matplotlib import gridspec
from sklearn.datasets import make_blobs

# Generate synthetic clusterable data using make_blobs
# 4 clusters, 30 samples, 10 features
data, cluster_labels = make_blobs(n_samples=20, n_features=10, centers=4, cluster_std=1.5, random_state=42)

# Calculate the distance matrix and apply hierarchical clustering
row_linkage = linkage(pdist(data, metric='euclidean'), method='average')
col_linkage = linkage(pdist(data.T, metric='euclidean'), method='average')

# Set up the matplotlib figure with gridspec
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(3, 3, width_ratios=[0.2, 1, 0.05], height_ratios=[0.2, 1, 0.05])

# Create axes for the row and column dendrograms, the heatmap, and the color bar
ax_row_dendrogram = fig.add_subplot(gs[1, 0], xticks=[], yticks=[], frame_on=False)
ax_col_dendrogram = fig.add_subplot(gs[0, 1], xticks=[], yticks=[], frame_on=False)
ax_heatmap = fig.add_subplot(gs[1, 1])
ax_cbar = fig.add_subplot(gs[1, 2])

# Row dendrogram (left)
dendro_row = dendrogram(row_linkage, ax=ax_row_dendrogram, orientation='left', no_labels=True)
ax_row_dendrogram.set_xticks([])
ax_row_dendrogram.set_yticks([])

# Column dendrogram (top)
dendro_col = dendrogram(col_linkage, ax=ax_col_dendrogram, orientation='top', no_labels=True)
ax_col_dendrogram.set_xticks([])
ax_col_dendrogram.set_yticks([])

# Reorder the data matrix based on clustering results
reordered_data = data[np.ix_(dendro_row['leaves'], dendro_col['leaves'])]

# Plot the heatmap with clustered data
sns.heatmap(reordered_data, ax=ax_heatmap, cmap='coolwarm', cbar=True, cbar_ax=ax_cbar,
            xticklabels=False, yticklabels=False)

# Place the color bar on the right and label it
ax_cbar.yaxis.set_ticks_position('right')
# ax_cbar.set_ylabel('Intensity', rotation=270, labelpad=15)

# Adjust the layout
plt.tight_layout()
plt.savefig('hirearchichal.png')
