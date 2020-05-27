import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn import metrics


train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\MLCup\\Internal Testing\\train-self.csv")


dbdata = train


X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]

##Performs the girdsearch
KNR = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=7)
parameters = {
#'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
#'leaf_size': (10,15,20, 25, 30, 35, 40,45,50),
#'metric': ('euclidean', 'manhattan', 'minkowski','chebyshev'),
'n_neighbors': (np.arange(1,30)),
#'p': (1, 2, 3, 5 , 10),
#'weights': ('uniform', 'distance')
 }


clf = GridSearchCV(KNR, parameters,scoring='neg_mean_squared_error', iid=False,cv=10,verbose=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))