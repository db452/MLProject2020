import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import validation_curve, GridSearchCV, RandomizedSearchCV
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


dbdata = train_encoded

    

# X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
# y_train = dbdata.iloc[:, 0]

X_train = train_encoded.iloc[:, np.arange(32)]
y_train = train_encoded.iloc[:, 32]

##Performs the girdsearch
knclassifier = KNeighborsClassifier(weights='uniform',metric='manhattan',algorithm='kd_tree',p=1,leaf_size=6,n_neighbors=37)

###Pass Params, default param grid
# 'n_neighbours':(1,2,3,4,5,6,7,8,9,10,20,30,50,100),
# 'weights':('uniform','distancer'),
# 'algorithm':('ball_tree','kd_tree','brute'),
# 'leaf_size':(1,5,10,20,30,50,60,70,80,90,100,200,500),
# 'p':(1,2,3,5,10),
# 'metric':('euclidean','manhattan','chebyshev','minkowski','mahalanobis')

parameters = {
    'n_neighbors':(np.arange(25,41)),
    'weights':('uniform','distance'),
    'algorithm':('ball_tree','kd_tree','brute'),
    'leaf_size':(np.arange(6,20)),
    'p':(1,2,3,5,10),
    'metric':('euclidean','manhattan','chebyshev','minkowski')

}



clf = GridSearchCV(knclassifier, parameters,scoring='accuracy', iid=False,cv=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('src/GridSearch/Results/KNNoutputstudent.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/KNNoutputstudent.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/KNNoutputstudent.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/KNNoutputstudent.log', 'a'))