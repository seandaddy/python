import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Generate the data
np.random.seed(0)
n_samples = 500

# Mean and covariance matrix
mean = [0, 0, 0]
cov = [
    [1, 0.8, -0.5],  # X highly correlated with Y, negatively with Z
    [0.8, 1, 0.2],  # Y less correlated with Z
    [-0.5, 0.2, 1],
]  # Z negatively correlated with X, less with Y

# Draw samples from the 3D Gaussian distribution
data = np.random.multivariate_normal(mean, cov, n_samples)
X, Y, Z = data[:, 0], data[:, 1], data[:, 2]

# Step 2: Apply PCA
pca = PCA(n_components=3)
pca.fit(data)

# Transform data into the new PCA space
data_pca = pca.transform(data)

# Step 3: Intuition and explanation
# The explained variance ratios
explained_variance = pca.explained_variance_ratio_

# Plot the original 3D data and the PCA-transformed data
fig = plt.figure(figsize=(12, 6))

# Original 3D data
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(X, Y, Z, alpha=0.5)
ax1.set_title("Original 3D Data")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# PCA-transformed data
ax2 = fig.add_subplot(122)
ax2.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
ax2.set_title("Data after PCA (First 2 Components)")
ax2.set_xlabel("First principal component")
ax2.set_ylabel("Second principal component")

plt.show()

# Print the explained variance
print("Explained variance ratio of the principal components:", explained_variance)
