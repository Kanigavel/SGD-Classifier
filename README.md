# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries such as pandas, sklearn.datasets, and SGDClassifier from sklearn.linear_model.

2.Load and split the dataset — use the Iris dataset and divide it into training and testing sets.

3.Train the model using the SGDClassifier and evaluate its accuracy on the test data.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Kanigavel M 
RegisterNumber: 212224240070


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SGD Classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train the classifier
sgd_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = sgd_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of SGD Classifier:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

 
*/
```

## Output:

<img width="897" height="386" alt="image" src="https://github.com/user-attachments/assets/c18a08c3-e3a6-41b2-8596-c2c8aa3d5f3b" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
