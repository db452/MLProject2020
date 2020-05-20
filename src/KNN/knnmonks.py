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

    

X_train = dbdata.iloc[:, [0,1,2,3,4,5]]
y_train = dbdata.iloc[:, 6]
X_test = dbdata2.iloc[:, [0,1,2,3,4,5]]
y_test = dbdata2.iloc[:, 6]


best = True

if best == True:
    #####Graphs the best results obtained from the gridsearch
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size= 2, n_neighbors= 5, p=1, weights = 'uniform', metric = 'manhattan')
    knn.fit(X_train,y_train)
    X_pred=knn.predict(X_train)
    y_pred=knn.predict(X_test)
else:
    ##Performs the girdsearch
    svclassifier = KNeighborsClassifier(n_jobs=-1)
    parameters = {'n_neighbors':(np.arange(1,25)),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree'),
    'leaf_size':(1,2,3,4,5,6,7,8,9,10,20,30,40),'p':(1,2,3,5,10),'metric':('euclidean','manhattan','minkowski')}##
    clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=10,verbose=10)
    clf.fit(X_train,y_train)  
    print(clf)
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_) 


##monk1={'algorithm': 'auto', 'leaf_size': 30, 'metric': 'euclidean', 'n_neighbors': 6, 'p': 1, 'weights': 'uniform'}TR/TS:87%,81%---MSE 0.12903,0.19444
##monks2={'algorithm': 'auto', 'leaf_size': 1, 'metric': 'euclidean', 'n_neighbors': 4, 'p': 1, 'weights': 'uniform'} TR/TS:73%,67%%---MSE 0.26627,0.32639
##monks3={'algorithm': 'ball_tree', 'leaf_size': 2, 'metric': 'manhattan', 'n_neighbors': 5, 'p': 1, 'weights': 'uniform'} TR/TS:92%,91---0.08197,0.09259
##algorithm=kd_tree, leaf_size=10, metric=minkowski, n_neighbors=6, p=3, weights=distance


########################
########GRAPH#######
########################
param_range = np.arange(1, 10, 1)
train_scores, test_scores = validation_curve(
    knn, X_train, y_train, param_name="n_neighbors", param_range=param_range,
    cv=3, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for KNN (MONKS3)")
plt.xlabel('N-neighbours')
plt.ylabel('Score')
lw = 2
plt.xlim(1,10)
plt.ylim(0,1)
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)

plt.legend(loc="best")


############################################

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print('Training Score is ', accuracy_score(y_train,X_pred))
print('Testing Score is ', accuracy_score(y_test,y_pred))
print('Training Error is ', mean_squared_error(y_train,X_pred))
print('Testing Error is ', mean_squared_error(y_test,y_pred))
plt.show()


