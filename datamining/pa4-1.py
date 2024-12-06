# import os
import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering

gravity = pd.read_csv('/Users/eer/Downloads/gravity80.csv')
gravity = gravity[gravity['year'] >= 2010]
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
# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(data)
## Plot the result
plt.scatter(pc.iloc[:,13], pc.iloc[:,12], c=dbscan.labels_, cmap='Paired')
plt.savefig('pa4-3.png')

# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Apply Spectral Clustering
spectral_clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=0)
spectral_labels = spectral_clustering.fit_predict(data)

# Scatter plot
plt.scatter(pc.iloc[:, 13], pc.iloc[:, 12], c=spectral_labels, cmap='coolwarm')
plt.title('Spectral Clustering')
plt.xlabel('Gravity')
plt.ylabel('Trade Volume')
plt.colorbar(label='Cluster Labels')
plt.savefig('pa4-5.png')
