import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
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

svm1 = SVC(coef0= 0, decision_function_shape= 'ovo', gamma='auto', kernel= 'poly', shrinking= True, tol=0.001,probability=True,C=35)
param_range1 = np.arange(1, 40)
title1='SVC Learning Curve for MONKS1'

svm2 = SVC(kernel='poly',class_weight='balanced',decision_function_shape='ovo',shrinking=False,coef0=2,degree=3,gamma='auto',tol=0.0005,probability=True, C=1000)#
param_range2 = np.arange(950,1001)
title2='SVC Learning Curve for MONKS2'

svm3 = SVC(class_weight='balanced',decision_function_shape='ovo',gamma='auto',kernel='poly',shrinking=True,coef0=1,tol=0.0001,probability=True,C=100)
param_range3 = np.arange(1,20)
title3='SVC Learning Curve for MONKS3'  



svm4=SVC(class_weight=None,decision_function_shape='ovo',gamma='scale',kernel='poly',shrinking=True,C=1, coef0=1, tol=0.0001)

title4='SVC Learning Curve for Students'  


def run(dbdata, dbdata2, svclassifier, param_range,title):
    global counter

    X_train = dbdata.iloc[:, np.arange(32)]
    y_train = dbdata.iloc[:, 32]
    X_test = dbdata2.iloc[:, np.arange(32)]
    y_test = dbdata2.iloc[:, 32]

    #####Graphs the best results obtained from the gridsearch
    svclassifier.fit(X_train, y_train)
    X_pred= svclassifier.predict(X_train)
    y_pred = svclassifier.predict(X_test)  

    ########################
    ########VALIDATION CURVE#######
    ########################

    # train_scores, test_scores = validation_curve(
    #     svclassifier, X_train, y_train, param_name="C", param_range=param_range,
    #     cv=3, scoring="accuracy", n_jobs=-1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)

    # plt.title("Accuracy Score of SVM(MONKS%d)" % counter)
    # plt.xlabel('C')
    # plt.ylabel("Accuracy")
    # lw = 2
    # plt.ylim(0,1)
    # plt.plot(param_range, train_scores_mean, label="Training score",
    #             color="darkorange", lw=lw)
    # plt.plot(param_range, test_scores_mean, label="Testing score",
    #             color="navy", lw=lw)

    # plt.legend(loc="best")
    ########################################
    ########################################
    #################LEARNING CURVE#########
    ########################################

    plot_learning_curve(svclassifier, title, X_train, y_train, ylim=(0, 1.01),cv=5, n_jobs=4)



    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))
    print('Training Score is ', accuracy_score(y_train,X_pred))
    print('Testing Score is ', accuracy_score(y_test,y_pred))
    print('Training Error is ', mean_squared_error(y_train,X_pred))
    print('Testing Error is ', mean_squared_error(y_test,y_pred))
    plt.savefig('img/SVM/Students%d' % counter)
    counter +=1
    plt.clf()


#run(monks1train,monks1test,svm1,param_range1,title1)


#run(monks2train,monks2test,svm2,param_range2,title2)


#run(monks3train,monks3test,svm3,param_range3,title3)

run(train_encoded,test_encoded,svm4,param_range3,title4)