import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, accuracy_score
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from learningcurve import  plot_learning_curve

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


counter = 1 

knn1 = KNeighborsClassifier(algorithm='auto',metric='chebyshev',weights='distance',n_neighbors=9,leaf_size=7)
param_range1 = np.arange(1, 30)
title1='Learning Curve MONKS1'

knn2 = KNeighborsClassifier(algorithm='brute', metric='chebyshev',weights='uniform',leaf_size=30,n_neighbors=1,p=1)
param_range2 = np.arange(1,10)
title2='Learning Curve MONKS2'

knn3 = KNeighborsClassifier(algorithm='auto',metric='manhattan',weights='distance',p=2,n_neighbors=9,leaf_size=60)
param_range3 = np.arange(1,40)
title3='Learning Curve MONKS3'

knnstudent = KNeighborsClassifier(weights='uniform',metric='manhattan',algorithm='kd_tree',p=1,leaf_size=6,n_neighbors=9)
title4='Learning Curve Students'

def run(dbdata, dbdata2, knn, param_range,title):
    global counter
    
    X_train = dbdata.iloc[:, np.arange(32)]
    y_train = dbdata.iloc[:, 32]
    X_test = dbdata2.iloc[:, np.arange(32)]
    y_test = dbdata2.iloc[:, 32]





    #####Graphs the best results obtained from the gridsearch
    knn.fit(X_train,y_train)
    X_pred=knn.predict(X_train)
    y_pred=knn.predict(X_test)




    ###############################
    ########Validation Curve#######
    ###############################
    # train_scores, test_scores = validation_curve(
    #     knn, X_train, y_train, param_name="n_neighbors", param_range=param_range,
    #     cv=5, scoring="neg_brier_score", n_jobs=-1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)

    # plt.title("Brier Score for KNN (Monks%d)" % counter)
    # plt.xlabel('N-neighbours')
    # plt.ylabel('Loss')
    # lw = 2
    # #plt.ylim(0,1)
    # plt.plot(param_range, train_scores_mean, label="Training score",
    #             color="darkorange", lw=lw)
    # plt.plot(param_range, test_scores_mean, label="Testing score",
    #             color="navy", lw=lw)

    # plt.legend(loc="best")


    ########################################
    #################Learning curve#########
    ########################################


    plot_learning_curve(knn, title, X_train, y_train, ylim=(0, 1.01),cv=5, n_jobs=4)



    ############################################

    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))
    print('Training Score is ', accuracy_score(y_train,X_pred))
    print('Testing Score is ', accuracy_score(y_test,y_pred))
    print('Training Error is ', mean_squared_error(y_train,X_pred))
    print('Testing Error is ', mean_squared_error(y_test,y_pred))
    plt.savefig('img/KNN/Student%d_learning' % counter)
    counter +=1
    plt.clf()

#run(monks1train,monks1test,knn1,param_range1,title1)
#run(monks2train,monks2test,knn2,param_range2,title2)
#run(monks3train,monks3test,knn3,param_range3,title3)
run(train_encoded,test_encoded,knnstudent,param_range1,title4)





