import pandas as pd  
import numpy as np  
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
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

dbdata = monks2train


X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
y_train = dbdata.iloc[:, 0]

##Performs the girdsearch
svclassifier = SVC()
parameters = {'C': [1,5,10,25,50,100,200],
'class_weight': (None, 'balanced'),
'coef0': [1,2,5,10],
'decision_function_shape': ('ovo', 'ovr'),
'gamma': (0, 0.1, 1, 'auto', 'scale'),
'kernel': ('rbf', 'sigmoid'),
'shrinking': (True, False),
'tol': (0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001)}


clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=5,n_jobs=-1,verbose=10)
clf.fit(X_train,y_train)  


print(clf,file=open('src/GridSearch/Results/SVMoutput.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/SVMoutput.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/SVMoutput.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/SVMoutput.log', 'a'))