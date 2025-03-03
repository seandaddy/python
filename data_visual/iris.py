# %%
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# %%
# Load the Iris dataset
iris = sns.load_dataset("iris")

# %%
# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(iris.drop(columns="species").corr(), annot=True, cmap="coolwarm")
plt.title("Heatmap of Iris Dataset")
plt.show()

# %%
# Pairwise Scatter Plots
sns.pairplot(iris, hue="species")
plt.suptitle("Pairwise Scatter Plots of Iris Dataset", y=1.02)
plt.show()

# %%
# Parallel Coordinates Plot
plt.figure(figsize=(12, 8))
parallel_coordinates(iris, "species", color=["blue", "orange", "green"])
plt.title("Parallel Coordinates Plot of Iris Dataset")
plt.show()

# %%
# Bubble Chart
plt.figure(figsize=(10, 8))
sizes = iris["sepal_length"] * 100  # Scale sepal length for bubble size
sns.scatterplot(
    data=iris,
    x="petal_length",
    y="petal_width",
    hue="species",
    size=sizes,
    sizes=(20, 200),
    alpha=0.5,
)
plt.title("Bubble Chart of Iris Dataset")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()
