import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
# Chi-squared distance
def chi_squared_distance(x1, x2):
    return np.sum((x1 - x2) ** 2 / (x1 + x2 + 1e-10))
class CustomKNN:
    def __init__(self, k=3):
        self.k = k   
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        predictions = []
        for sample in X:
            distances = [chi_squared_distance(sample, x) for x in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
# Load data
iris = load_iris()
X, y = iris.data, iris.target
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Train & test
knn = CustomKNN(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
