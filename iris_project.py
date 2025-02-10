# -----------------------------------------
# Author: Daniel Saravia
# Class: SWE 452
# Description: Load and analyze the Iris dataset using scikit-learn
# -----------------------------------------

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
iris = load_iris()

# Features (input data)
X = iris.data

# Target labels
y = iris.target

# Feature and target names
feature_names = iris.feature_names
target_names = iris.target_names

# Display dataset details
print("Feature names:\n", feature_names)
print("Target names:\n", target_names)
print("First 5 samples:\n", X[:5])

# Convert to DataFrame
data = pd.DataFrame(iris.data, columns=feature_names)

# Display missing values per feature
print("\nMissing values per feature:\n", data.isnull().sum())

# Display number of duplicate rows
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame for better readability
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

# Display first 5 rows of the scaled training set
print("\nFirst 5 rows of the scaled training data:\n", X_train_scaled_df.head())

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Display classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()