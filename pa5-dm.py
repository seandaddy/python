# %%
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS, RandomEffects,PooledOLS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP

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
# Standardize the data
scaler = StandardScaler()
data = pd.DataFrame( scaler.fit_transform( df.iloc[:,3:] ) , columns=df.columns[3:] )
# Dimension reduction for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
pca_data = pd.DataFrame(pca_data, columns=['PC1','PC2'])

# %%
reconstructed_data = pca.inverse_transform(pca_data)
spe = np.sum((data - reconstructed_data)**2, axis=1)

# Set threshold
threshold = np.percentile(spe, 90)

# Identify outliers
outliers = np.where(spe > threshold)[0]

# Plot
plt.scatter(pca_data['PC1'], pca_data['PC2'])
plt.scatter(pca_data.loc[outliers, 'PC1'], pca_data.loc[outliers, 'PC2'], c='r')
plt.show()

# %%
print(outliers)
print(len(outliers))

# %%
pca_data = pd.concat([df[['year']], pca_data], axis=1)
pca_data.head()

# %%
pca_loadings = pd.DataFrame(pca.components_, columns=df.columns[3:], index=['PC1','PC2'])
pca_loadings

# %%
# Plot the two PC vectors as arrows
plt.figure(figsize=(8, 6))
plt.scatter(pca_data["PC1"], pca_data["PC2"], c=pca_data['year'])
plt.plot([0, pca.components_[0, 0]*5], [0, pca.components_[0, 1]*5], color='red', label='PC1')
plt.plot([0, pca.components_[1, 0]*4], [0, pca.components_[1, 1]*4], color='blue', label='PC2')
if pca.components_.shape[0] > 2:
    plt.plot([0, pca.components_[2, 0]*3], [0, pca.components_[2, 1]*3], color='pink', label='PC3')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid()
plt.show()

# %%
# Truncated SVD
svd = TruncatedSVD(n_components=2)
svd_data = svd.fit_transform(data)
svd_data = pd.DataFrame(svd_data, columns=['PC1','PC2'])
svd_data = pd.concat([df[['year']], svd_data], axis=1)
svd_data.head()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(svd_data['PC1'], svd_data['PC2'], c=svd_data['year'])
plt.title('Truncated SVD')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()

# %%
# NMF
nmf = NMF(n_components=2)
## NMF does not work with negative values
nmf_data = nmf.fit_transform(data + abs(data.min().min()))
nmf_data = pd.DataFrame(nmf_data, columns=['PC1','PC2'])
nmf_data = pd.concat([df[['year']], nmf_data], axis=1)
nmf_data.head()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(nmf_data['PC1'], nmf_data['PC2'], c=nmf_data['year'])
plt.title('NMF')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()

# %%
# ICA
ica = FastICA(n_components=2)
ica_data = ica.fit_transform(data)
ica_data = pd.DataFrame(ica_data, columns=['PC1','PC2'])
ica_data = pd.concat([df[['year']], ica_data], axis=1)
ica_data.head()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(ica_data['PC1'], ica_data['PC2'], c=ica_data['year'])
plt.title('ICA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()


# %%
# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda_data = lda.fit_transform(data, df['year'])
lda_data = pd.DataFrame(lda_data, columns=['PC1','PC2'])
lda_data = pd.concat([df[['year']], lda_data], axis=1)
lda_data.head()

# %%
plt.figure(figsize=(8, 6))
plt.scatter(lda_data['PC1'], lda_data['PC2'], c=lda_data['year'])
plt.title('LDA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()

# %%
print(data.shape)
print(pca_data.shape)
print(svd_data.shape)
print(nmf_data.shape)
print(ica_data.shape)
print(lda_data.shape)

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
