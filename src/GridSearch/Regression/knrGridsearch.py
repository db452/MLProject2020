import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import LabelEncoder


student_train = pd.read_csv("src/dataset/Students/student-train.csv", header=None)

train_encoded = student_train.apply(LabelEncoder().fit_transform)

#train = pd.read_csv("src/dataset/MLCup/Internal Testing/train-self.csv")


dbdata = train_encoded


X_train = dbdata.iloc[:, np.arange(32)]
y_train = dbdata.iloc[:, 32]

##Performs the girdsearch
KNR = KNeighborsRegressor()
parameters = {
'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
'leaf_size': (10,15,20, 25, 30, 35, 40,45,50),
'metric': ('euclidean', 'manhattan', 'minkowski','chebyshev'),
'n_neighbors': (np.arange(1,30)),
'p': (1, 2, 3, 5 , 10),
'weights': ('uniform', 'distance')
}


clf = GridSearchCV(KNR, parameters,scoring='neg_mean_squared_error', iid=False,cv=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('src/GridSearch/Results/KNRoutputstudent.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/KNRoutputstudent.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/KNRoutputstudent.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/KNRoutputstudent.log', 'a'))