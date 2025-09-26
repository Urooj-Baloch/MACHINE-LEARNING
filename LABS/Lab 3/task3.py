import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- Load Dataset -------------------
df = pd.read_csv("cancer patient data sets.csv")

# A) Exploratory Data Analysis
print("First look at the dataset:")
print(df.head())
print(df.info())
print(df.describe())

# B) Basic Data Checks
class_distribution = df["Level"].value_counts(normalize=True)
missing_values = df.isnull().sum()
duplicate_count = df.duplicated().sum()
categorical_cols = df.select_dtypes(include=["object", "category"])

print("\nClass distribution (target = Level):\n", class_distribution)
print("\nTotal missing values:", missing_values.sum())
print("Duplicate rows:", duplicate_count)
print("\nCategorical columns:\n", categorical_cols)

# Dataset looks clean → no missing values, no duplicates.
# Patient Id and index are identifiers → safe to remove.
# Target column "Level" is categorical → needs encoding.
# Class distribution is fairly balanced.
df = df.drop(['Patient Id', 'index'], axis=1)

# Encode target labels (Low, Medium, High → 0, 1, 2)
encoder = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
df["Level"] = encoder.fit_transform(df[["Level"]])

# ------------------- Correlation -------------------
corr_matrix = df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print("Correlation Table:\n", corr_matrix)

# Histograms → quick view of feature distributions
df.hist(figsize=(12, 12))
plt.tight_layout()
plt.show()

# Scaling is important for KNN since features have different ranges
X = df.drop("Level", axis=1)
y = df["Level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- Train-Test Splits -------------------
# Split 1 → 80% training, 20% testing
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)

# Split 2 → 70% training, 30% validation
X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
    X_scaled, y, test_size=0.3, random_state=0
)

# Why use a validation set?
# It helps tune hyperparameters (like k or metric) without touching the final test set.

# ------------------- KNN with Different Distance Metrics -------------------
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
results = {}

# Using k=15 as an example
for metric in distance_metrics:
    knn = KNeighborsClassifier(n_neighbors=15, metric=metric)
    knn.fit(X_train_a, y_train_a)

    pred_test = knn.predict(X_test_a)
    pred_val = knn.predict(X_val_b)

    acc_test = accuracy_score(y_test_a, pred_test)
    acc_val = accuracy_score(y_val_b, pred_val)

    results[metric] = {
        'test_accuracy': acc_test,
        'val_accuracy': acc_val
    }

# results
print("\nAccuracy comparison across distance metrics:")
results_df = pd.DataFrame(results).T
print(results_df)

# Observation:
# Scaling the features was important because KNN relies on distance.
# It is noticed that accuracy changes depending on the distance metric used.
# In this dataset, Manhattan distance gave a small advantage,
# but the best choice can vary with different data.
