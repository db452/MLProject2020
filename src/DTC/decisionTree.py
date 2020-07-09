import pandas as pd  
import numpy as np 

 
import matplotlib.pyplot as plt  
from sklearn.metrics import  accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
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

dtc1 = DecisionTreeClassifier(splitter= 'random', random_state= 3, min_weight_fraction_leaf= 0.1, min_samples_split= 2, min_samples_leaf= 10, min_impurity_decrease= 0, max_leaf_nodes= 5, max_features= None, max_depth= None, criterion= 'gini', class_weight= None)
param_range1 = np.arange(1, 20)
title1='DTC Learning curve for MONKS1'

dtc2 = DecisionTreeClassifier(splitter= 'random', random_state= 3, min_weight_fraction_leaf= 0, min_samples_split= 2, min_samples_leaf= 1, min_impurity_decrease= 0, max_leaf_nodes= None, max_features= None, max_depth= 10, criterion= 'gini', class_weight= 'balanced')#
param_range2 = np.arange(1,20)
title2='DTC Learning curve for MONKS1'

dtc3 = DecisionTreeClassifier(splitter= 'random', random_state= 1, min_weight_fraction_leaf= 0, min_samples_split= 0.1, min_samples_leaf= 1, min_impurity_decrease= 0, max_leaf_nodes= None, max_features= None, max_depth= None, criterion= 'gini', class_weight= None)
param_range3 = np.arange(1,20)
title3='DTC Learning curve for MONKS3'

dtc4=DecisionTreeClassifier(splitter='random',min_impurity_decrease=0,max_features=None,criterion='gini',class_weight=None,min_weight_fraction_leaf=0,min_samples_split=0.1,max_leaf_nodes=13,max_depth=191,min_samples_leaf=4)

title4='DTC Learning curve for Students'

def run(dbdata, dbdata2, DTclassifier, param_range,title):
    global counter
    
    X_train = dbdata.iloc[:, [1,2,3,4,5,6]]#np.arange(32
    y_train = dbdata.iloc[:, 0]#32

    X_test = dbdata2.iloc[:, [1,2,3,4,5,6]]#np.arange(32
    y_test = dbdata2.iloc[:, 0]#32




    ######
    


    DTclassifier.fit(X_train, y_train)
    X_pred= DTclassifier.predict(X_train)
    y_pred = DTclassifier.predict(X_test)  

    ########################
    ########VALIDATION CURVE#######
    ########################
    # train_scores, test_scores = validation_curve(
    #     DTclassifier, X_train, y_train, param_name="random_state", param_range=param_range,
    #     cv=3, scoring="neg_brier_score", n_jobs=-1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)

    # plt.title("Brier Score of DTC(MONKS%d)" % counter)
    # plt.xlabel('Random State')
    # plt.ylabel("Loss")
    # lw = 2
    # #plt.ylim(0,1)
    # plt.plot(param_range, train_scores_mean, label="Training score",
    #             color="darkorange", lw=lw)
    # plt.plot(param_range, test_scores_mean, label="Testing score",
    #             color="navy", lw=lw)

    # plt.legend(loc="best")
    # ##################################
    ########################################
    ########################################
    #################LEARNING CURVE#########
    ########################################


    plot_learning_curve(DTclassifier, title, X_train, y_train, ylim=(0, 1.01),cv=5, n_jobs=4)


    print('Training Score is ', accuracy_score(y_train,X_pred))
    print('Testing Score is ', accuracy_score(y_test,y_pred))
    print('Training Error is ', mean_squared_error(y_train,X_pred))
    print('Testing Error is ', mean_squared_error(y_test,y_pred))
    plt.savefig('img/DTC/Students%d' % counter)
    counter +=1
    plt.clf()


# run(monks1train,monks1test,dtc1,param_range1,title1)


# run(monks2train,monks2test,dtc2,param_range2,title2)


# run(monks3train,monks3test,dtc3,param_range3,title3)

run(train_encoded,test_encoded,dtc4,param_range3,title4)