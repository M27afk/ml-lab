def minkowski(Q, R, p):
    '''
    This function calculates the distance between 2 vectors
    p=1 will give Manhattan distance
    p=2 will give Euclidean distance
    '''
    distance = (np.sum(np.abs(Q - R) ** p) ** (1 / p))
    return distance

import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNearestNeighbors:
    def __init__(self, k=3, p_metric=2):
        self.k = k
        self.p_metric = p_metric

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_Pred):
        Y_Pred = [self.classify(x) for x in X_Pred]
        return np.array(Y_Pred)

    def classify(self, x):
        distances = [minkowski(x, x_train, self.p_metric) for x_train in self.X_train]
        idx_k = np.argsort(distances)[:self.k]
        k_labels = [self.Y_train[i] for i in idx_k]
        predicted_class = Counter(k_labels).most_common(1)[0][0]
        return predicted_class

# Load the Glass dataset
glass = pd.read_csv("glass.csv", skiprows=1)

# Remove the last column from X and keep only the last column for y
X = glass.iloc[:, :-1].values
y = glass.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of KNearestNeighbors with k=3 and p_metric=2
knn = KNearestNeighbors(k=3, p_metric=2)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
