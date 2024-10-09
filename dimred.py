# %%
import tensorflow as tf
with tf.device("/device:GPU:0"):
    pass

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP
import matplotlib.pyplot as plt
# %%
gravity = pd.read_csv('~/Documents/python/data/gravity80.csv')
#gravity = gravity[gravity['year']>1990]
df = gravity
df['trade'] = df['tradeflow_comtrade_o'] + df['tradeflow_comtrade_d']
# df['rhsg'] = (df['gdp_o'] * df['gdp_d'])/ (df['dist'])**2
df['gatt'] = df['gatt_o'] * df['gatt_d']
df['wto'] = df['wto_o'] * df['wto_d']
df['eu'] = df['eu_o'] * df['eu_d']
df = df.drop(columns=['gatt_o', 'gatt_d', 'wto_o', 'wto_d', 'eu_o', 'eu_d', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d'])
# df = df.drop(columns=['dist', 'gatt_o', 'gatt_d', 'wto_o', 'wto_d', 'eu_o', 'eu_d', 'gdp_o', 'gdp_d', 'tradeflow_comtrade_o', 'tradeflow_comtrade_d'])
df.head()

# %%
# Normalize
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(df.iloc[:, 3:]), columns=df.columns[3:])

# %%
# Applying t-SNE
tsne = TSNE(n_components=2, perplexity=30, max_iter=250)
tsne_data = tsne.fit_transform(data)
tsne_data = pd.DataFrame(tsne_data, columns=['PC1','PC2'])
tsne_data = pd.concat([df[['year']], tsne_data], axis=1)
tsne_data.head()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(tsne_data['PC1'], tsne_data['PC2'], c=tsne_data['year'])
plt.title('t-SNE')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()

# %%
# Applying UMAP
umap = UMAP(n_components=2)
umap_data = umap.fit_transform(data)
umap_data = pd.DataFrame(umap_data, columns=['PC1','PC2'])
umap_data = pd.concat([df[['year']], umap_data], axis=1)
umap_data.head()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(umap_data['PC1'], umap_data['PC2'], c=umap_data['year'])
plt.title('UMAP')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()
