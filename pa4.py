# import os
import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

gravity = pd.read_csv('/Users/eer/Documents/python/data/gravity80.csv')

df = gravity
df['trade'] = df['tradeflow_comtrade_o'] + df['tradeflow_comtrade_d']
df['rhsg'] = (df['gdp_o'] * df['gdp_d'])/ (df['dist'])**2
# Standardize the data
scaler = StandardScaler()
data = pd.DataFrame( scaler.fit_transform( df.iloc[:,3:] ) , columns=df.columns[3:])

# Dimension reduction for visualization (we will cover it in the next class)
pca = PCA()
pc = pd.DataFrame(pca.fit_transform(data))

# Create a figure and set of subplots with a 3x2 layout
fig, axes = plt.subplots(3, 2, figsize=(12, 15))  # Adjust size as needed
axes = axes.flatten()  # Flatten the 3x2 grid to easily iterate over it

# Define the variable combinations to plot
combinations = [(0, 10), (0, 11), (1, 10), (1, 11), (2, 10), (2, 11)]

# Iterate through the combinations and plot
for i, (var_x, var_y) in enumerate(combinations):
    ax = axes[i]  # Get the current axis
    scatter = ax.scatter(pc.iloc[:, var_x], pc.iloc[:, var_y], c=df['year'], cmap='viridis')
    
    ax.set_title(f'Scatter Plot of Variable {var_x} vs Variable {var_y}')
    ax.set_xlabel(f'Variable {var_x}')  # Label for x-axis
    ax.set_ylabel(f'Variable {var_y}')  # Label for y-axis

    # Add colorbar for each subplot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Year')

# Adjust layout
plt.tight_layout()
plt.savefig('pa4-1.png')

fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust size as needed
axes = axes.flatten()  # Flatten the 3x3 grid to easily iterate over it

# Iterate over the desired number of clusters
for n_clusters in range(2, 11):  # 2 to 11 inclusive
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    
    # Plotting the clusters
    ax = axes[n_clusters - 2]  # Adjust index for subplot
    scatter = ax.scatter(pc.iloc[:, 13], pc.iloc[:, 12], c=kmeans.labels_, cmap='Paired')
    
    ax.set_title(f'K-Means Clustering with n_clusters={n_clusters}')
    ax.set_xlabel('Gravity')  # Adjust labels as necessary
    ax.set_ylabel('Trade Volume')
    ax.legend(*scatter.legend_elements(), title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()
plt.savefig('pa4-2.png')

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(data)
## Plot the result
plt.scatter(pc.iloc[:,13], pc.iloc[:,12], c=dbscan.labels_, cmap='Paired')
plt.savefig('pa4-3.png')

# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=5, random_state=0)  # You can change the number of components
gmm_labels = gmm.fit_predict(data)

# Scatter plot
plt.scatter(pc.iloc[:, 12], pc.iloc[:, 13], c=gmm_labels, cmap='plasma')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Gravity')
plt.ylabel('Trade Volume')
plt.colorbar(label='Cluster Labels')
plt.savefig('pa4-4.png')

# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Apply Spectral Clustering
spectral_clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=0)
spectral_labels = spectral_clustering.fit_predict(data)

# Scatter plot
plt.scatter(pc.iloc[:, 12], pc.iloc[:, 13], c=spectral_labels, cmap='coolwarm')
plt.title('Spectral Clustering')
plt.xlabel('Gravity')
plt.ylabel('Trade Volume')
plt.colorbar(label='Cluster Labels')
plt.savefig('pa4-5.png')
