# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Training Logistic Regression and Support Vector Machines using Scikit-Learn

import numpy as np
import pandas as pd
from sklearn import datasets

# +
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')


# -

# ## First steps with scikit-learn
#
# Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower samples. The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

# +
iris = datasets.load_iris()
#print(iris)
X = iris.data[:,[0,2]]
#X = iris.data
y = iris.target

print('Class labels:', np.unique(y))
print(X.shape)

#####      MINE        ########
df = pd.read_csv("diabetes_data_upload.csv")
print(data.shape)
df.head()




# -

# ## Encode categorical features
#
#

# +
labels = {'Gender': {'Male': 1, 'Female': 0}, 'class': {'Positive': 1, 'Negative': 0}, 'others': {'Yes': 1, 'No': 0}}
catCols = df.select_dtypes("object").columns

for col in catCols:
    label = labels.get(col, labels['others'])
    
    # to convert label from strings to integers
    df[col] = df[col].map(label)
    
df.head()
# -

# # IGNORE EVERYTHING BELOW 

# ## Get X and y

data = df.to_numpy()
X = data[:, 0:16]
y = data[:, 16]
print(X)
print(y)

# Splitting data into 70% training and 30% test data:

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.28, random_state=None, stratify=y)
print(X_train.shape)
print(X_test.shape)
# -

print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

print(X[::10, :])

# ## Standardizing features

# +
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
#sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# -

print(X_train_std[::10, :])

# ## Training a logistic regression with scikit-learn

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1)
lr.fit(X_train_std, y_train)

# ## Check the performance of the model

# +
# Calculate the score of training data

score_train = lr.score(X_train_std, y_train)
print("Training score = ",score_train)

score_test = lr.score(X_test_std, y_test)
print("Test score = ", score_test)
# -

print(lr.predict(X_train_std[::10, :]))
print(lr.predict_proba(X_train_std[::10, :]))
print(y_train[::10])

# ## Parameters of the model

print("Coef: ", lr.coef_)
print("Intercept: ", lr.intercept_)
print("n_iter: ", lr.n_iter_)

print(X.shape)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()

# ## Training Support Vector Machines with scikit-learn

from sklearn.svm import SVC
svc = SVC(C=1000, probability=True, kernel='linear')
svc.fit(X_train_std, y_train)

# ## Check the performance of the model

# +
# Calculate the score of training data

score_train = svc.score(X_train_std, y_train)
print("Training score = ",score_train)

score_test = svc.score(X_test_std, y_test)
print("Test score = ", score_test)
# -

print(svc.predict(X_train_std[::10, :]))
print(svc.predict_proba(X_train_std[::10, :]))
print(y_train[::10])

# ## Parameters of the model

print("n_support: ", svc.n_support_)

print(X.shape)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()




