import pandas as pd  
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

##import official train and test
train = pd.read_csv("src\\dataset\\MLCup\\train.csv", header=None)
test = pd.read_csv("src\\dataset\\MLCup\\test-Official.csv",header=None) 


dbdata = train
dbdata2 = test

#### format data for input
X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 20]
X_test = dbdata2.iloc[:, np.arange(20)]


##set KNN with best params
knn = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=15)


##fit and predict data
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

###Format for the second column


y_train2 = dbdata.iloc[:, 21]


###train with new parameters

knn2 = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=7)


##fit and predict data
knn2.fit(X_train,y_train2)
y_pred2=knn2.predict(X_test)


pd.DataFrame(y_pred2,y_pred).to_csv('src/dataset/MLCup/columns.csv')



#####figure out where last value disappears to