import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder

student_train = pd.read_csv("src/dataset/Students/student-train.csv", header=None)

train_encoded = student_train.apply(LabelEncoder().fit_transform)


#train = pd.read_csv("src/dataset/MLCup/Internal Testing/train-self.csv")


dbdata = train_encoded


X_train = dbdata.iloc[:, np.arange(32)]
y_train = dbdata.iloc[:, 32]

##Performs the girdsearch
svregressor = SVR()
parameters = {
'kernel': ('rbf', 'poly', 'sigmoid'),
'C': (1.0, 10, 100, 1000),
'coef0': (0.0, 0.5, 1, 10),
'gamma': (0, 0.1, 1, 'auto', 'scale'),
'shrinking': (True, False),
'tol': (0.0001, 1e-06, 0.01)
}


clf = GridSearchCV(svregressor, parameters,scoring='neg_mean_squared_error', iid=False,cv=5,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('src/GridSearch/Results/SVRoutputstudent.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/SVRoutputstudent.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/SVRoutputstudent.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/SVRoutputstudent.log', 'a'))