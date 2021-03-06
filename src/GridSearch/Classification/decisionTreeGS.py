import pandas as pd  
import numpy as np  
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
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

X_train = monks1train.iloc[:, [1,2,3,4,5,6]]#np.arange(32
y_train = monks1train.iloc[:, 0]#32



dtClassifier = DecisionTreeClassifier()

# Original Hyperparameters tested
# 'criterion':('gini', 'entropy'),
# 'splitter': ('best','random'),
# 'max_depth':(None,10,100,1000),
# 'min_samples_split':(0.1,0.2,0.3,0.4,0.5,0.7,0.9,2,10,50,100),
# 'min_samples_leaf':(1,5,10),
# 'min_weight_fraction_leaf':(0,0.1,0.2,0.3,0.4,0.5),
# 'max_features':(None,'auto','sqrt','log2'),
# 'max_leaf_nodes':(None,5,10,100),
# 'min_impurity_decrease':(0.0,0.5,1.0),
# 'class_weight':(None,'balanced')



parameters = {
'criterion':('gini', 'entropy'),
'splitter': ('best','random'),
'max_depth':(None,10,100,1000),
'min_samples_split':(0.1,0.2,0.3,0.4,0.5,0.7,0.9,2,10,50,100),
'min_samples_leaf':(1,5,10),
'min_weight_fraction_leaf':(0,0.1,0.2,0.3,0.4,0.5),
'max_features':(None,'auto','sqrt','log2'),
'max_leaf_nodes':(None,5,10,100),
'min_impurity_decrease':(0,0.5,1.0),
'class_weight':(None,'balanced'),
'random_state':(1,2,3,4,5)
}


clf = RandomizedSearchCV(dtClassifier, parameters,scoring='accuracy', iid=False,cv=5,n_jobs=-1,n_iter=10000)#n_iter=1000
clf.fit(X_train,y_train)



print(clf,file=open('src/GridSearch/Results/DTCoutputmonks1.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/DTCoutputmonks1.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/DTCoutputmonks1.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/DTCoutputmonks1.log', 'a'))

