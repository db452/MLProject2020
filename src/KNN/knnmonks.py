import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import validation_curve, GridSearchCV

monks1train = pd.read_csv("src/dataset/Monk/pandasdataset/monks1train.csv")
monks1test = pd.read_csv("src/dataset/Monk/pandasdataset/monks1test.csv")  

monks2train = pd.read_csv("src/dataset/Monk/pandasdataset/monks2train.csv")
monks2test = pd.read_csv("src/dataset/Monk/pandasdataset/monks2test.csv")  

monks3train = pd.read_csv("src/dataset/Monk/pandasdataset/monks3train.csv")
monks3test = pd.read_csv("src/dataset/Monk/pandasdataset/monks3test.csv")  


counter = 1 

knn1 = KNeighborsClassifier(algorithm='auto',metric='chebyshev',weights='distance',n_neighbors=28,leaf_size=7)
param_range1 = np.arange(1, 30)

knn2 = KNeighborsClassifier(algorithm='brute', metric='mahalanobis',weights='uniform',leaf_size=30,n_neighbors=1,p=1)
param_range2 = np.arange(1,10)

knn3 = KNeighborsClassifier(algorithm='auto',metric='manhattan',weights='distance',p=2,n_neighbors=30,leaf_size=60)
param_range3 = np.arange(1,40)


def run(dbdata, dbdata2, knn, param_range):
    global counter
    
    X_train = dbdata.iloc[:, [1,2,3,4,5,6]]
    y_train = dbdata.iloc[:, 0]
    X_test = dbdata2.iloc[:, [1,2,3,4,5,6]]
    y_test = dbdata2.iloc[:, 0]





    #####Graphs the best results obtained from the gridsearch
    knn.fit(X_train,y_train)
    X_pred=knn.predict(X_train)
    y_pred=knn.predict(X_test)




    ########################
    ########GRAPH#######
    ########################
    train_scores, test_scores = validation_curve(
        knn, X_train, y_train, param_name="n_neighbors", param_range=param_range,
        cv=3, scoring="accuracy", n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve for KNN (Monks%d)" % counter)
    plt.xlabel('N-neighbours')
    plt.ylabel('Accuracy Score')
    lw = 2
    plt.ylim(0,1)
    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Testing score",
                color="navy", lw=lw)

    plt.legend(loc="best")


    ############################################

    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))
    print('Training Score is ', accuracy_score(y_train,X_pred))
    print('Testing Score is ', accuracy_score(y_test,y_pred))
    print('Training Error is ', mean_squared_error(y_train,X_pred))
    print('Testing Error is ', mean_squared_error(y_test,y_pred))
    plt.savefig('img/KNN/monks%d' % counter)
    counter +=1
    plt.clf()

run(monks1train,monks1test,knn1,param_range1)
run(monks2train,monks2test,knn2,param_range2)
run(monks3train,monks3test,knn3,param_range3)


