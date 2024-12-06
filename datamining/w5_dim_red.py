# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from umap import UMAP
import matplotlib.pyplot as plt

# %%
# Loading the history data
df = pd.read_csv("~/Documents/python/data/DAAN545_Seshat.csv")

# %%
# Normalize
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(df.iloc[:, 3:]), columns=df.columns[3:])

# %%
# Try different dimension reduction techniques and visualize in 2D
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
pca_data = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
pca_data = pd.concat([df[["NGA", "Time"]], pca_data], axis=1)
pca_data.head()
axes[0, 0].scatter(pca_data["PC1"], pca_data["PC2"], c=pca_data["Time"])
axes[0, 0].set_title("PCA")

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel="rbf")
kpca_data = kpca.fit_transform(data)
kpca_data = pd.DataFrame(kpca_data, columns=["PC1", "PC2"])
kpca_data = pd.concat([df[["NGA", "Time"]], kpca_data], axis=1)
kpca_data.head()
axes[0, 1].scatter(kpca_data["PC1"], kpca_data["PC2"], c=kpca_data["Time"])
axes[0, 1].set_title("Kernel PCA")

# Truncated SVD
svd = TruncatedSVD(n_components=2)
svd_data = svd.fit_transform(data)
svd_data = pd.DataFrame(svd_data, columns=["PC1", "PC2"])
svd_data = pd.concat([df[["NGA", "Time"]], svd_data], axis=1)
svd_data.head()
axes[0, 2].scatter(svd_data["PC1"], svd_data["PC2"], c=svd_data["Time"])
axes[0, 2].set_title("Truncated SVD")

# NMF
nmf = NMF(n_components=2)
## NMF does not work with negative values
nmf_data = nmf.fit_transform(data + abs(data.min().min()))
nmf_data = pd.DataFrame(nmf_data, columns=["PC1", "PC2"])
nmf_data = pd.concat([df[["NGA", "Time"]], nmf_data], axis=1)
nmf_data.head()
axes[1, 0].scatter(nmf_data["PC1"], nmf_data["PC2"], c=nmf_data["Time"])
axes[1, 0].set_title("NMF")

# ICA
ica = FastICA(n_components=2)
ica_data = ica.fit_transform(data)
ica_data = pd.DataFrame(ica_data, columns=["PC1", "PC2"])
ica_data = pd.concat([df[["NGA", "Time"]], ica_data], axis=1)
ica_data.head()
axes[1, 1].scatter(ica_data["PC1"], ica_data["PC2"], c=ica_data["Time"])
axes[1, 1].set_title("ICA")

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda_data = lda.fit_transform(data, df["Time"])
lda_data = pd.DataFrame(lda_data, columns=["PC1", "PC2"])
lda_data = pd.concat([df[["NGA", "Time"]], lda_data], axis=1)
lda_data.head()
axes[1, 2].scatter(lda_data["PC1"], lda_data["PC2"], c=lda_data["Time"])
axes[1, 2].set_title("LDA")

plt.show()
# %%
