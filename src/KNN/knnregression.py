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
monks1test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Regressiontest\\winequalitytest.csv")  

dbdata = monks1train
dbdata2 = monks1test

    

X_train = dbdata.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]]
y_train = dbdata.iloc[:, 11]
X_test = dbdata2.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]]
y_test = dbdata2.iloc[:, 11]



knn = KNeighborsRegressor(algorithm='auto', leaf_size= 1, n_neighbors= 10, p=1, weights = 'distance', metric = 'manhattan')
knn.fit(X_train,y_train)
X_pred=knn.predict(X_train)
y_pred=knn.predict(X_test)



print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))
