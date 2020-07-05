import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder

monks1train = pd.read_csv("src/dataset/Monk/pandasdataset/monks1train.csv")
monks1test = pd.read_csv("src/dataset/Monk/pandasdataset/monks1test.csv")  

monks2train = pd.read_csv("src/dataset/Monk/pandasdataset/monks2train.csv")
monks2test = pd.read_csv("src/dataset/Monk/pandasdataset/monks2test.csv")  

monks3train = pd.read_csv("src/dataset/Monk/pandasdataset/monks3train.csv")
monks3test = pd.read_csv("src/dataset/Monk/pandasdataset/monks3test.csv")  

student_train = pd.read_csv("src/dataset/Students/student-train.csv", header=None)
student_test = pd.read_csv("src/dataset/Students/student-test.csv", header=None)



train_encoded = student_train.apply(LabelEncoder().fit_transform)

dbdata = train_encoded


X_train = dbdata.iloc[:, np.arange(32)]
y_train = dbdata.iloc[:, 32]

##Performs the girdsearch
svclassifier = SVC(class_weight=None,decision_function_shape='ovo',gamma='scale',kernel='poly',shrinking=True,C=1, coef0=1, tol=0.0001)
parameters = {'C': (np.arange(1,25)),
#'class_weight': (None, 'balanced'),
'coef0': (np.arange(1,10)),
#'decision_function_shape': ('ovo', 'ovr'),
#'gamma': (0, 0.1, 1, 'auto', 'scale'),
#'kernel': ('rbf', 'poly', 'linear', 'sigmoid'),
#'shrinking': (True, False),
'tol': (0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001)}


clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=5,verbose=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('src/GridSearch/Results/SVMoutputstudent.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/SVMoutputstudent.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/SVMoutputstudent.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/SVMoutputstudent.log', 'a'))