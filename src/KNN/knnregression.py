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
test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\MLCup\\Internal Testing\\test-self.csv")  

dbdata = train
dbdata2 = test

    

X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]
X_test = dbdata2.iloc[:, np.arange(20)]
y_test = dbdata2.iloc[:, 21]



knn = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=7)
knn.fit(X_train,y_train)
X_pred=knn.predict(X_train)
y_pred=knn.predict(X_test)



print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))
