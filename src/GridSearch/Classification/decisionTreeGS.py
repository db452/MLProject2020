import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


student_train = pd.read_csv("src/dataset/Students/student-train.csv", header=None)
student_test = pd.read_csv("src/dataset/Students/student-test.csv", header=None)



train_encoded = student_train.apply(LabelEncoder().fit_transform)
test_encoded = student_test.apply(LabelEncoder().fit_transform)

X_train = train_encoded.iloc[:, np.arange(32)]
y_train = train_encoded.iloc[:, 32]



dtClassifier = DecisionTreeClassifier()

parameters = {
'criterion':('gini', 'entropy'),
'splitter': ('best','random'),
'max_depth':(None,10,100,1000),
'min_samples_split':(1,2,10,50,100),
'min_samples_leaf':(1,5,10),
'min_weight_fraction_leaf':(0,1,2),
'max_features':(None,'auto','sqrt','log2'),
'max_leaf_nodes':(None,5,10,100),
'min_impurity_decrease':(0,0.5,1.0),
'class_weight':(None,'balanced')
}


clf = RandomizedSearchCV(dtClassifier, parameters,scoring='accuracy', iid=False,cv=5,verbose=10,n_jobs=-1)
clf.fit(X_train,y_train)



print(clf,file=open('src/GridSearch/Results/DTCoutput.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/DTCoutput.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/DTCoutput.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/DTCoutput.log', 'a'))

