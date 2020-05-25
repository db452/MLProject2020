import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score

monks1train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks1train.csv")
monks1test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks1test.csv")  

monks2train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks2train.csv")
monks2test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks2test.csv")  

monks3train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks3train.csv")
monks3test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks3test.csv")  

dbdata = monks3train
dbdata2 = monks3test


X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
y_train = dbdata.iloc[:, 0]

##Performs the girdsearch
svclassifier = SVC(max_iter=-1,class_weight='balanced',decision_function_shape='ovo',gamma='auto',kernel='poly',shrinking=True,C=10,coef0=1,tol=0.0001)
parameters = {'C': (1, 10, 100, 1000),
'class_weight': ('None', 'balanced'),
'coef0': (0, 0.5, 1, 10),
'decision_function_shape': ('ovo', 'ovr'),
'gamma': (0, 0.1, 1, 'auto', 'scale'),
'kernel': ('rbf', 'poly', 'linear', 'sigmoid'),
'shrinking': (True, False),
'tol': (0.0001, 1e-06, 0.01)}


clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=5,verbose=10)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.log', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.log', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.log', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.log', 'a'))