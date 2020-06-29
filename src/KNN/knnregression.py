import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn import metrics

train = pd.read_csv("src\\dataset\\MLCup\\Internal Testing\\train-self.csv")
test = pd.read_csv("src\\dataset\\MLCup\\Internal Testing\\test-self.csv")  

dbdata = train
dbdata2 = test

    

X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]
X_test = dbdata2.iloc[:, np.arange(20)]
y_test = dbdata2.iloc[:, 21]



knn = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=7)
knn.fit(X_train,y_train)
X_pred=knn.predict(X_train)
y_pred=knn.predict(X_test)



########################
########GRAPH#######
########################
param_range = np.arange(1,30)
train_scores, test_scores = validation_curve(
    knn, X_train, y_train, param_name="n_neighbors", param_range=param_range,
    cv=5, scoring="explained_variance", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for KNR")
plt.xlabel('Number of Nearest Neighbors')
plt.ylabel('Explained Variance Score(Y)')
lw = 2
plt.ylim(0.95,1)
plt.plot(param_range, train_scores_mean, label="Training score",
            color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Testing score",
            color="navy", lw=lw)

plt.legend(loc="best")




print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))

plt.savefig('img/KNR/KNR_Y.png')
