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
test_encoded = student_test.apply(LabelEncoder().fit_transform)


dbdata = monks3train
dbdata2 = monks3test


X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
y_train = dbdata.iloc[:, 0]
X_test = dbdata2.iloc[:, [1,2,3,4,5,6]]
y_test = dbdata2.iloc[:, 0]

  



#####Graphs the best results obtained from the gridsearch
svclassifier = SVC(class_weight=None,decision_function_shape='ovo',gamma='scale',kernel='poly',shrinking=True,C=1, coef0=1, tol=0.0001)
svclassifier.fit(X_train, y_train)
X_pred= svclassifier.predict(X_train)
y_pred = svclassifier.predict(X_test)  



##best params monks1:{'C': 810, 'coef0': 0, 'decision_function_shape': 'ovo', 'gamma': 0.005, 'kernel': 'rbf', 'shrinking': True, 'tol': 0.000001}}
##best params monks2:{'C': 1000, 'coef0': 0, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.0001}
##best params monks3{'C': 0.5, 'coef0': 0, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'rbf', 'shrinking': True, 'tol': 0.1}
##{'C': 10, 'coef0': 1, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'poly', 'shrinking': True, 'tol': 0.0001}

########################
########GRAPH#######
########################

param_range = np.arange(1,20)
train_scores, test_scores = validation_curve(
    svclassifier, X_train, y_train, param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Effects of Regularization parameter on SVM(MONKS1)")
plt.xlabel('C')
plt.ylabel("Accuracy")
lw = 2
#plt.xlim()
plt.ylim(0,1)
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Testing score",
             color="navy", lw=lw)

plt.legend(loc="best")
##################################

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print('Training Score is ', accuracy_score(y_train,X_pred))
print('Testing Score is ', accuracy_score(y_test,y_pred))
print('Training Error is ', mean_squared_error(y_train,X_pred))
print('Testing Error is ', mean_squared_error(y_test,y_pred))
plt.savefig('img/SVM/svmMonk3.png')