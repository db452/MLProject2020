import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn import metrics


train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Regressiontest\\winequalitytrain.csv")
test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Regressiontest\\winequalitytest.csv")  

dbdata = train
dbdata2 = test


X_train = dbdata.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]]
y_train = dbdata.iloc[:, 11]
X_test = dbdata2.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]]
y_test = dbdata2.iloc[:, 11]


svrclassifier = SVR(kernel='rbf',C=100, coef0=1, gamma= 0.7, shrinking= True, tol=0.1)  
svrclassifier.fit(X_train, y_train)
X_pred= svrclassifier.predict(X_train)
y_pred = svrclassifier.predict(X_test)  




##print('Training Score is ', metrics.euclidean_distances(y_train,X_pred))
##print('Testing Score is ', metrics.euclidean_distances(y_test,y_pred))


print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))
