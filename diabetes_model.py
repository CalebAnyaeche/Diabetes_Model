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

# # Early Stage Diabetes Risk Prediction Model

# Team Members: Caleb Anyaeche and Matthew Zlibut

# #### Background 
#
# Diabetes is one of the fastest-growing chronic life-threatening diseases that have already affected 422 million people worldwide according to the report of the World Health Organization (WHO), in 2018. Because of the presence of a generally long asymptomatic stage, early detection of diabetes is constantly wanted for a clinically significant result. Around 50% of all people suffering from diabetes are undiagnosed because of its long-term asymptomatic phase. The early diagnosis of diabetes is only possible by proper assessment of both common and less common sign symptoms, which could be found in different phases from disease initiation up to diagnosis. Data mining classification techniques have been well accepted by researchers for the risk prediction model of the disease.

# #### Goal
#
# The objective is to build a machine learning based model to predict if a patient has or will have a early stage diabetes risk.

# #### Reference Paper Result
#
# the best result was achieved using Random Forest Algorithm where using tenfold cross-validation 97.4% instances were classified correctly and using percentage split technique, it could classify 99% of the instances correctly.

# ## Dataset Information:
#

# #### Dependent Attribute
#
# Class (1.Positive, 2.Negative)

# #### Independent Attribute
#
# Age (20-65)
# Sex (1.Male, 2.Female)
# Polyuria (1.Yes, 2.No)
# Polydipsia (1.Yes, 2.No)
# Sudden Weight Loss (1.Yes, 2.No)
# Weakness (1.Yes, 2.No)
# Polyphagia (1.Yes, 2.No)
# Genital Thrush (1.Yes, 2.No)
# Visual Blurring (1.Yes, 2.No)
# Itching (1.Yes, 2.No)
# Irritability (1.Yes, 2.No)
# Delayed Healing (1.Yes, 2.No)
# Partial Paresis (1.Yes, 2.No)
# Muscle Stiffness (1.Yes, 2.No)
# Alopecia (1.Yes, 2.No)
# Obesity (1.Yes, 2.No)

# #### Missing Data
#
# The dataset has no missing data.

# #### Dropped Columns 
#
# No column needs to be dropped, as neither of them are keys nor can be derived from others.

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix

# ## Load Dataset
#
# Loading the Diabetes dataset.

df = pd.read_csv("diabetes_data_upload.csv")
df.head()

# ## Encode categorical features
#
#

# Categorical data is coverted to integer label. 1=Male, 0=Female; 1=Positive, 0=Negative; 1=Yes, 0=No.

# +
labels = {'Gender': {'Male': 1, 'Female': 0}, 'class': {'Positive': 1, 'Negative': 0}, 'others': {'Yes': 1, 'No': 0}}
catCols = df.select_dtypes("object").columns

for col in catCols:
    label = labels.get(col, labels['others'])
    
    # to convert label from strings to integers
    df[col] = df[col].map(label)
    
df.head()
# -

# ## Get X and y

# Here, the first sixteenth column represents the independent variable, while the seventeenth column represents the dependent variable.

# +
data = df.to_numpy()
X = data[:, 0:16]
y = data[:, 16]

print(X.shape)
print('Class labels:', np.unique(y))
# -

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
X_test_std = sc.transform(X_test)
# -

print(X_train_std[::10, :])

# ## Training a logistic regression with scikit-learn

# +
from sklearn.linear_model import LogisticRegression

#setting up the hyperparameter grid
param_grid = [{'C': np.logspace(-4, 2, 7)}]
lr = LogisticRegression()
#using gridsearch cross validation in order to find the best hyperparameters
gs = GridSearchCV(estimator=lr, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=10,
    n_jobs=-1)
gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
# -

# ## Check the performance of the model

# +
# Calculate the score of training and testing data
print('Accuracy training: ', gs.best_score_)
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
print('Accuracy test: ', accuracy_score(y_test, y_pred_test))

print('Precision: ', precision_score(y_train, y_pred_train))
print('Recall: ', recall_score(y_train, y_pred_train))
print('F1: ', f1_score(y_train, y_pred_train))

#the confusion matrix for training and testing data
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

print(gs.best_estimator_.predict(X_train_std[::10, :]))
print(gs.best_estimator_.predict_proba(X_train_std[::10, :]))
print(y_train[::10])

# ### Parameters of the model

print("Coef: ", gs.best_estimator_.coef_)
print("Intercept: ", gs.best_estimator_.intercept_)
print("n_iter: ", gs.best_estimator_.n_iter_)

# ## Training Support Vector Machines with scikit-learn

# +
from sklearn.svm import SVC

#setting the range for c and gamma
C_range = np.logspace(-4, 2, 7)
gamma_range = np.logspace(-4, 2, 7)

#setting up the parameters with their respective kernels
param_grid = [{'C': C_range, 'kernel': ['linear']},
    {'C': C_range, 
    'gamma': gamma_range, 
    'kernel': ['rbf']}]

svc = SVC()
#setting up the gridsearch cross validation
gs = GridSearchCV(estimator=svc, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=10,
    n_jobs=-1)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
# -

# ## Check the performance of the model

# +
# Calculate the scores of training and testing data
print('Accuracy: ', gs.best_score_)

y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
print('Accuracy test: ', accuracy_score(y_test, y_pred_test))
print('Precision: ', precision_score(y_train, y_pred_train))
print('recall: ', recall_score(y_train, y_pred_train))
print('f1: ', f1_score(y_train, y_pred_train))

#confusion matrix for training and testing
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

print(gs.best_estimator_.predict(X_train_std[::10, :]))
# print(gs.best_estimator_.predict_proba(X_train_std[::10, :]))
print(y_train[::10])

# ### Parameters of the model

print("n_support: ", gs.best_estimator_.n_support_)

# ## Training Multi-Layer Perceptron Classifier with scikit-learn

# +
#making the hidden layer range that is used
hls_range = [(m,n) for m in range(8,3,-2) for n in range(m,3,-2)]+[(n,) for n in range(8,3,-1)]
print(hls_range)
#making a range of alpha values for the model
alpha_range = np.logspace(-2,2,5)
print(alpha_range)
#creating the parameter grid
param_grid = [{'alpha':alpha_range, 'hidden_layer_sizes':hls_range}]

#setting up the model
gs = GridSearchCV(estimator=MLPClassifier(tol=1e-5, 
                                          learning_rate_init=0.02,
                                          max_iter=1000,
                                         random_state=1), 
                  param_grid=param_grid, 
                  cv=5)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)

# +
#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)
print("The accuracy for the training data is :", gs.best_estimator_.score(X_train_std,y_train))
print("The accuracy for the test data is :",gs.best_estimator_.score(X_test_std,y_test))
#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -


