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
from sklearn.model_selection import cross_val_score

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
lr = gs.best_estimator_
# -

# ## Check the performance of the model

# +
# Calculate the score of training and testing data
lr_t_acc = gs.best_score_
print('Accuracy training: ', lr_t_acc)
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
lr_acc = accuracy_score(y_test, y_pred_test)
lr_prec = precision_score(y_train, y_pred_train)
lr_recall = recall_score(y_train, y_pred_train)
lr_f1 = f1_score(y_train, y_pred_train)
print('Accuracy test: ', lr_acc)
print('Precision: ', lr_prec)
print('Recall: ', lr_recall)
print('F1: ', lr_f1)

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
svc = gs.best_estimator_
# -

# ## Check the performance of the model

# +
# Calculate the scores of training and testing data
svm_t_acc = gs.best_score_
print('Accuracy: ', svm_t_acc)

y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

svm_acc = accuracy_score(y_test, y_pred_test)
svm_prec = precision_score(y_train, y_pred_train)
svm_recall = recall_score(y_train, y_pred_train)
svm_f1 = f1_score(y_train, y_pred_train)

print('Accuracy test: ', svm_acc)
print('Precision: ', svm_prec)
print('recall: ', svm_recall)
print('f1: ', svm_f1)

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
mlp = gs.best_estimator_

# +
#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)

mlp_t_acc =  gs.best_estimator_.score(X_train_std,y_train)
mlp_acc = gs.best_estimator_.score(X_test_std,y_test)
mlp_prec = precision_score(y_train, y_pred_train)
mlp_recall = recall_score(y_train, y_pred_train)
mlp_f1 = f1_score(y_train, y_pred_train)

print("The accuracy for the training data is :", mlp_t_acc)
print("The accuracy for the test data is :", mlp_acc)
print('Precision: ', mlp_prec)
print('recall: ', mlp_recall)
print('f1: ', mlp_f1)

#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# ## Decision Tree

# +
from sklearn.tree import DecisionTreeClassifier

param_grid=[{'max_depth': [5, 6, 7, 8, 9]}]

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
dt = gs.best_estimator_

# +
#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)

dt_t_acc = gs.best_estimator_.score(X_train_std,y_train)
dt_acc = gs.best_estimator_.score(X_test_std,y_test)
dt_prec = precision_score(y_train, y_pred_train)
dt_recall = recall_score(y_train, y_pred_train)
dt_f1 = f1_score(y_train, y_pred_train)

print("The accuracy for the training data is :", dt_t_acc)
print("The accuracy for the test data is :", dt_acc)
print('Precision: ', dt_prec)
print('recall: ', dt_recall)
print('f1: ', dt_f1)

#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# ## Random Forest Classifier

# +
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

param_gird={
    'max_depth': [8, 9, 10],
}

gs = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
rfc = gs.best_estimator_

# +
#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)

rf_t_acc = gs.best_estimator_.score(X_train_std,y_train)
rf_acc = gs.best_estimator_.score(X_test_std,y_test)
rf_prec = precision_score(y_train, y_pred_train)
rf_recall = recall_score(y_train, y_pred_train)
rf_f1 = f1_score(y_train, y_pred_train)

print("The accuracy for the training data is :", rf_t_acc)
print("The accuracy for the test data is :", rf_acc)
print('Precision: ', rf_prec)
print('recall: ', rf_recall)
print('f1: ', rf_f1)

#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# # Ensemble with Majority Voting classifier

# +
clf1 = LogisticRegression(C=0.1)
clf2 = SVC(C=10.0, gamma=0.1)
clf3 = MLPClassifier(alpha=0.01, hidden_layer_sizes=(8, 4), learning_rate_init=0.02, max_iter=1000, random_state=1,tol=1e-05)
clf4 = DecisionTreeClassifier(max_depth=5, random_state=0)
clf5 = RandomForestClassifier(max_depth=6, random_state=1)

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('mlp', clf3), ('dt', clf4), ('rf', clf5)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)

el_t_acc = eclf1.score(X_train, y_train)
el_acc = eclf1.score(X_test, y_test)
el_prec = precision_score(y_train, y_pred_train)
el_recall = recall_score(y_train, y_pred_train)
el_f1 = f1_score(y_train, y_pred_train)

print("The accuracy for the training data is :", el_t_acc)
print("The accuracy for the test data is :", el_acc)
y_pred_train = eclf1.predict(X_train)
print('Precision: ', el_prec)
print('recall: ', el_recall)
print('f1: ', el_f1)

# -

# # Result Summary
# Accuracy yield from different models. The best result was achieved using Random Forest Algorithm with an accuracy of 98%. 

# +
import matplotlib.pyplot as plt
# %matplotlib inline

models = ["LR", "SVM", "MLP", "DT", "RF", "EL"]
accuracy = [a*100 for a in [lr_acc, svm_acc, mlp_acc, dt_acc, rf_acc, el_acc]]
incorrect = [100-a for a in accuracy]
xpos = np.arange(len(models))

plt.xticks(xpos, models)
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Performance of Classification Algorithms")
plt.bar(xpos, accuracy, label="Correctly classified instances")
plt.bar(xpos, incorrect, label="Incorrectly classified instances")
plt.legend(loc="best")
ax = plt.gca()
ax.set_ylim([0,130])
# -
models_dict = {'Linear Regression': [lr_t_acc, lr_acc, lr_prec, lr_recall, lr_f1],
               'Support Vector Classification': [svm_t_acc, svm_acc, svm_prec, svm_recall, svm_f1], 
               'Multi-Layered Perceptron': [mlp_t_acc, mlp_acc, mlp_prec, mlp_recall, mlp_f1], 
               'Decision Tree': [dt_t_acc, dt_acc, dt_prec, dt_recall, dt_f1], 
               'Random Forest': [rf_t_acc, rf_acc, rf_prec, rf_recall, rf_f1], 
               'Majority Vote': [el_t_acc, el_acc, el_prec, el_recall, el_f1]}

# +
models_df = pd.DataFrame.from_dict(models_dict).rename(index={0:'Training Accuracy', 1:'Testing Accuracy', 2:'Precision', 3:'Recall', 4:'F1'})

models_df
# -



