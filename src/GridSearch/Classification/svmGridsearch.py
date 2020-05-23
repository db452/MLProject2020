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

dbdata = monks1train
dbdata2 = monks1test


X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
y_train = dbdata.iloc[:, 0]

##Performs the girdsearch
svclassifier = SVC()
parameters = {'C':(0.25,0.5,0.75,1,10,100,1000,10000), 'kernel':('rbf','sigmoid'), 'gamma': (0.00001,0.0001,0.001,0.01,0.1,0.5,1,'auto'),'coef0':(0,1,2,3),
'shrinking':(True,False), 'tol':(0.0001,0.0005,0.001,0.005,0.01,0.1),'decision_function_shape':('ovo','ovr'),'max_iter':(-1,1,2,3,5,10,20,30)}#


clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=5,verbose=10)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.txt', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.txt', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.txt', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVMoutput.txt', 'a'))