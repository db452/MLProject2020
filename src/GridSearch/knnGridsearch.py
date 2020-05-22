import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import validation_curve, GridSearchCV

monks1train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks1train.csv")
monks1test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks1test.csv")  

monks2train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks2train.csv")
monks2test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks2test.csv")  

monks3train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks3train.csv")
monks3test = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\Monk\\pandasdataset\\monks3test.csv")  

dbdata = monks1train
dbdata2 = monks1test

    

X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
y_train = dbdata.iloc[:, 0]

##Performs the girdsearch
svclassifier = KNeighborsClassifier(n_jobs=-1)
parameters = {'n_neighbors':(np.arange(1,25)),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree'),
'leaf_size':(1,2,3,4,5,6,7,8,9,10,20,30,40),'p':(1,2,3,5,10),'metric':('euclidean','manhattan','minkowski')}##
clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=10,verbose=10)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNNoutput.txt', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNNoutput.txt', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNNoutput.txt', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\KNNoutput.txt', 'a'))