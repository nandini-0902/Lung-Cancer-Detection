import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn import tree

print("Dataset:")
dataset = pd.read_csv('lung_cancer_examples.csv')
print(len(dataset))
print(dataset.head())

scatter_matrix(dataset)
plt.show()

A = dataset[dataset.Result == 1]
B = dataset[dataset.Result == 0]

plt.scatter(A.Age, A.Smokes, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Smokes, color="Blue", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Smokes")
plt.legend()
plt.title("Smokes vs Age")
plt.show()

plt.scatter(A.Age, A.Alcohol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Alcohol, color="Blue", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Alcohol vs Age")
plt.show()

plt.scatter(A.Smokes, A.Alcohol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Smokes, B.Alcohol, color="Blue", label="0", alpha=0.4)
plt.xlabel("Smokes")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Alcohol vs Smokes")
plt.show()

# Split dataset
X = dataset.iloc[:, 3:5]
y = dataset.iloc[:, 6]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Feature Scaling
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print("--Using KNN Algorithm--")


a = math.sqrt(len(y_train))
print(a)

# Defining a model - KNN
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')

# Fit model
classifier.fit(x_train, y_train)

# Predict the test set result
y_predict = classifier.predict(x_test)
print(y_predict)

# Evaluate model
# Confusion matrix
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix: ")
print(cm)

print("In Confusion Matrix:-----")
print("Position 1.1 shows the patients that don't have Cancer, In this case =", cm[0, 0])
print("Position 1.2 shows the number of patients that have higher risk of Cancer, In this case =", cm[0, 1])
print("Position 2.1 shows the Incorrect Value, In this case =", cm[1, 0])
print("Position 2.2 shows the correct number of patients that have Cancer, In this case =", cm[1, 1])

print('F1 Score:', f1_score(y_test, y_predict))
print('Accuracy:', accuracy_score(y_test, y_predict))

# Using Decision tree
print("----")
print("Using Decision Tree Algorithm")

c = tree.DecisionTreeClassifier()
c.fit(x_train, y_train)

accu_train = np.sum(c.predict(x_train) == y_train) / float(y_train.size)
accu_test = np.sum(c.predict(x_test) == y_test) / float(y_test.size)

print('Classification accuracy on train', accu_train * 100)
print('Classification accuracy on test', accu_test * 100)
