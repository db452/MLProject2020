import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics

train = pd.read_csv("src\\dataset\\MLCup\\Internal Testing\\train-self.csv")
test = pd.read_csv("src\\dataset\\MLCup\\Internal Testing\\test-self.csv")  

dbdata = train
dbdata2 = test

    

X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]
X_test = dbdata2.iloc[:, np.arange(20)]
y_test = dbdata2.iloc[:, 21]



knn = KNeighborsRegressor(algorithm='ball_tree', metric='manhattan', weights='distance',p=10,n_neighbors=22,leaf_size=10)
knn.fit(X_train,y_train)
X_pred=knn.predict(X_train)
y_pred=knn.predict(X_test)



print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))
