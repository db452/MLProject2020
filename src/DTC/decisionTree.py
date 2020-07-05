import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
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
test_encoded = student_test.apply(LabelEncoder().fit_transform)

X_train = monks1train.iloc[:, [1,2,3,4,5,6]]#np.arange(32
y_train = monks1train.iloc[:, 0]#32

X_test = monks1test.iloc[:, [1,2,3,4,5,6]]#np.arange(32
y_test = monks1test.iloc[:, 0]#32




######
DTclassifier = DecisionTreeClassifier(class_weight= 'balanced', criterion= 'gini', max_depth= None, max_features= None, max_leaf_nodes= 10, min_impurity_decrease= 0, min_samples_leaf= 1, min_samples_split= 2, min_weight_fraction_leaf= 0, splitter= 'random',random_state=1)



DTclassifier.fit(X_train, y_train)
X_pred= DTclassifier.predict(X_train)
y_pred = DTclassifier.predict(X_test)  

print('Training Score is ', accuracy_score(y_train,X_pred))
print('Testing Score is ', accuracy_score(y_test,y_pred))
print('Training Error is ', mean_squared_error(y_train,X_pred))
print('Testing Error is ', mean_squared_error(y_test,y_pred))