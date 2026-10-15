import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from pathlib import Path

# Create a folder for saving charts
output_dir = Path("iris_charts")
output_dir.mkdir(exist_ok=True)

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Rename columns for easier understanding
df = df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    "target": "species_id"
})

# Convert species numbers into names
species_names = dict(enumerate(iris.target_names))
df["species"] = df["species_id"].map(species_names)

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nSpecies counts:")
print(df["species"].value_counts())

sns.set_theme(style="whitegrid")

# 1. BAR CHART: Average petal length by species
avg_petal = df.groupby("species", observed=True)["petal_length"].mean()

plt.figure(figsize=(8, 5))
avg_petal.plot(kind="bar")
plt.title("Average Petal Length by Iris Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / "bar_chart.png", dpi=300)
plt.show()

# 2. HISTOGRAM: Distribution of sepal length
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="sepal_length", bins=15, kde=True)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Number of Flowers")
plt.tight_layout()
plt.savefig(output_dir / "histogram.png", dpi=300)
plt.show()

# 3. SCATTER PLOT: Petal length vs petal width
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species",
    s=70
)
plt.title("Petal Length vs Petal Width by Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig(output_dir / "scatter_plot.png", dpi=300)
plt.show()

# Compare average measurements by species
print("\nAverage measurements by species:")
print(
    df.groupby("species")[
        ["sepal_length", "sepal_width",
         "petal_length", "petal_width"]
    ].mean().round(2)
)

print("\nAll charts saved in the iris_charts folder.")
