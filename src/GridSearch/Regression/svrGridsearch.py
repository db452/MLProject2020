import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score


monks1train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Regressiontest\\winequalitytrain.csv")


dbdata = monks1train


X_train = dbdata.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]]
y_train = dbdata.iloc[:, 11]

##Performs the girdsearch
svregressor = SVR(kernel='rbf',shrinking=True, coef0=0,gamma='scale',tol=0.001,C=200,epsilon=0.2)
parameters = {'C': (1, 10, 100, 1000), 
'coef0': (0, 1, 2, 3),
'epsilon': (0.1, 0.01, 1),
'gamma': (0.1, 0.5, 0.7, 1, 2, 3, 'auto', 'scale'),
'kernel': ('rbf', 'sigmoid', 'sigmoid'),
'shrinking': (True, False),
'tol': (0.001, 0.01, 0.0001)}# 


clf = GridSearchCV(svregressor, parameters,scoring='neg_mean_squared_error', iid=False,cv=10,verbose=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))