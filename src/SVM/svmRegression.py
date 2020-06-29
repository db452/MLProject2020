import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn import metrics


train = pd.read_csv("src/dataset/MLCup/Internal Testing/train-self.csv")
test = pd.read_csv("src/dataset/MLCup/Internal Testing/test-self.csv")  

dbdata = train
dbdata2 = test


X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]
X_test = dbdata2.iloc[:, np.arange(20)]
y_test = dbdata2.iloc[:, 21]


svrclassifier = SVR(coef0=0,kernel='rbf',shrinking=True,C=8,tol=0.003,epsilon=0.29,gamma=0.23)
svrclassifier.fit(X_train, y_train)
X_pred= svrclassifier.predict(X_train)
y_pred = svrclassifier.predict(X_test)  



########################
########GRAPH#######
########################
param_range = np.arange(1,20)
train_scores, test_scores = validation_curve(
    svrclassifier, X_train, y_train, param_name="C", param_range=param_range,
    cv=5, scoring="explained_variance", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve for SVR")
plt.xlabel('C')
plt.ylabel('Explained Variance Score(Y)')
lw = 2
plt.ylim(0,1)
plt.plot(param_range, train_scores_mean, label="Training score",
            color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Testing score",
            color="navy", lw=lw)

plt.legend(loc="best")




##print('Training Score is ', metrics.euclidean_distances(y_train,X_pred))
##print('Testing Score is ', metrics.euclidean_distances(y_test,y_pred))


print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))

plt.savefig('img/SVR/SVR_Y.png')
