import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error 
from sklearn.model_selection import validation_curve, GridSearchCV

dbdata = pd.read_csv("C://Users//user//Desktop//MLProject//indians.csv")  

######Preprocessing of data, refer to report

median_plglcconc = dbdata.loc[:,"Plasma glucose conc"].median()
median_bloodp = dbdata.loc[:,"Blood Pressure"].median()
median_skinthick = dbdata.loc[:,"Triceps thick"].median()
median_twohourserins = dbdata.loc[:,"serum insulin"].median()
median_bmi = dbdata.loc[:,"BMI"].median()

dbdata.replace(dbdata.loc[dbdata['Plasma glucose conc'] == 0], median_plglcconc )
dbdata.replace(dbdata.loc[dbdata['Blood Pressure'] == 0], median_bloodp )
dbdata.replace(dbdata.loc[dbdata['Triceps thick'] == 0], median_skinthick )
dbdata.replace(dbdata.loc[dbdata['serum insulin'] == 0], median_twohourserins )
dbdata.replace(dbdata.loc[dbdata['BMI'] == 0], median_bmi )
########################

X = dbdata.drop('Class', axis=1)  
y = dbdata['Class']
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)

best = True

if best == True:
    #####Graphs the best results obtained from the gridsearch
    knn = KNeighborsClassifier(algorithm='auto', leaf_size= 1, n_neighbors= 18, p=1, weights = 'uniform', metric = 'manhattan')
    knn.fit(X_train,y_train)
    X_pred=knn.predict(X_train)
    y_pred=knn.predict(X_test)
else:
    ##Performs the girdsearch
    svclassifier = KNeighborsClassifier(n_jobs=2)
    parameters = {'n_neighbors':(np.arange(1,25)),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree'),
    'leaf_size':(1,2,3,4,5,6,7,8,9,10,20,30,40),'p':(1,2,3,5,10),'metric':('euclidean','manhattan','minkowski')}##
    clf = GridSearchCV(svclassifier, parameters,scoring='accuracy', iid=False,cv=10,verbose=10)
    clf.fit(X_train,y_train)  
    print(clf)
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_) 
##best params:{'algorithm': 'auto', 'leaf_size': 1, 'metric': 'manhattan', 'n_neighbors': 18, 'p': 1, 'weights': 'uniform'} TR/TS:76%/77%---MSE 0.23836,0.22511

########################
########GRAPH#######
########################

param_range = np.arange(1, 20, 1)
train_scores, test_scores = validation_curve(
    knn, X_train, y_train, param_name="n_neighbors", param_range=param_range,
    cv=3, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for Indians")
plt.xlabel('N_neighbours')
plt.ylabel("Score")
lw = 2
plt.xlim(1,20)
plt.ylim(0,1)
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)

plt.legend(loc="best")

###############################

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print('Training Score is ', accuracy_score(y_train,X_pred))
print('Testing Score is ', accuracy_score(y_test,y_pred))
print('Training Error is ', mean_squared_error(y_train,X_pred))
print('Testing Error is ', mean_squared_error(y_test,y_pred))
plt.show()


