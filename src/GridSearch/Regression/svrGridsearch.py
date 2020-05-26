import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score


train = pd.read_csv("C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\dataset\\MLCup\\Internal Testing\\train-self.csv")


dbdata = train


X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]

##Performs the girdsearch
svregressor = SVR(coef0=0,kernel='rbf',shrinking=True,C=8,tol=0.003,epsilon=0.29,gamma=0.23)
parameters = {
}


clf = GridSearchCV(svregressor, parameters,scoring='neg_mean_squared_error', iid=False,cv=5,verbose=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))
print(clf.best_estimator_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))
print(clf.best_score_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))
print(clf.best_params_,file=open('C:\\Users\\Dennis\\Documents\\GitHub\\Machine-Learning2020\\src\\GridSearch\\Results\\SVRoutput.log', 'a'))