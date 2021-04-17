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
print('Accuracy training: ', gs.best_score_)
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
lr_acc = accuracy_score(y_test, y_pred_test)
print('Accuracy test: ', lr_acc)

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
svc = gs.best_estimator_
# -

# ## Check the performance of the model

# +
# Calculate the scores of training and testing data
print('Accuracy: ', gs.best_score_)

y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
svm_acc = accuracy_score(y_test, y_pred_test)
print('Accuracy test: ', svm_acc)
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
mlp = gs.best_estimator_

# +
#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)
print("The accuracy for the training data is :", gs.best_estimator_.score(X_train_std,y_train))
mlp_acc = gs.best_estimator_.score(X_test_std,y_test)
print("The accuracy for the test data is :", mlp_acc)
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
print("The accuracy for the training data is :", gs.best_estimator_.score(X_train_std,y_train))
dt_acc = gs.best_estimator_.score(X_test_std,y_test)
print("The accuracy for the test data is :", dt_acc)
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
print("The accuracy for the training data is :", gs.best_estimator_.score(X_train_std,y_train))
rf_acc = gs.best_estimator_.score(X_test_std,y_test)
print("The accuracy for the test data is :", rf_acc)
#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# # Majority Vote

# +
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
# from sklearn.externals import six
import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


# -

my_models = [lr, svc, mlp, dt, rfc]
mv_clf = MajorityVoteClassifier(classifiers=my_models)
all_models=[lr, svc, mlp, dt, rfc, mv_clf]
clf_labels = ['Logistic Regression', 'Support Vector Classification', 'Multi-Layer Perceptron', 
              'Decision Tree', 'Random Forest Classification', 'Majority Vote Classification']

# +
# le = LabelEncoder()

# y_e = le.fit_transform(y)

X_train, X_test, y_train, y_test =\
       train_test_split(X, y, 
                        test_size=0.5, 
                        random_state=1,
                        stratify=y)

# +
# for clf, label in zip(all_models, clf_labels):
#     scores = cross_val_score(estimator=mv_clf,
#                              X=X_train,
#                              y=y_train,
#                              cv=10,
#                              scoring='roc_auc')
#     print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
#           % (scores.mean(), scores.std(), label))
    
mv_clf.fit(X_train,y_train)
# -

y_pred = mv_clf.predict(X_test)
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred))

# # Ensemble with Majority Voting classifier

# +
clf1 = LogisticRegression(C=0.1)
clf2 = SVC(C=10.0, gamma=0.1)
clf3 = MLPClassifier(alpha=0.01, hidden_layer_sizes=(8, 4), learning_rate_init=0.02, max_iter=1000, random_state=1,tol=1e-05)
clf4 = DecisionTreeClassifier(max_depth=5, random_state=0)
clf5 = RandomForestClassifier(max_depth=6, random_state=1)

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('mlp', clf3), ('dt', clf4), ('rf', clf5)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
print("The accuracy for the training data is :", eclf1.score(X_train, y_train))
el_acc = eclf1.score(X_test, y_test)
print("The accuracy for the test data is :", el_acc)

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
ax.set_ylim([0,150])
# -




