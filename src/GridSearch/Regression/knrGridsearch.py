import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn import metrics


monks1train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Regressiontest\\winequalitytrain.csv")


dbdata = monks1train


X_train = dbdata.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]]
y_train = dbdata.iloc[:, 11]

##Performs the girdsearch
KNR = KNeighborsRegressor()
parameters = {'n_neighbors':(np.arange(10,25)),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree'),
'leaf_size':(1,2,3,4,5),'p':(1,2,3,5),'metric':('euclidean','manhattan','minkowski')}##


clf = GridSearchCV(KNR, parameters,scoring='explained_variance', iid=False,cv=5,verbose=10)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNRoutput.log', 'a'))