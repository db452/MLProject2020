import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn import metrics
from learningcurve import  plot_learning_curve
from euclidean_distances import euclidean



train = pd.read_csv("src\\dataset\\MLCup\\Model Testing\\train-self.csv",header=None)
test = pd.read_csv("src\\dataset\\MLCup\\Model Testing\\test-self.csv",header=None)  

dbdata = train
dbdata2 = test


X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 20]
X_test = dbdata2.iloc[:, np.arange(20)]
y_test = dbdata2.iloc[:, 20]


svrclassifier = SVR(shrinking=True,kernel='rbf',C=15,epsilon=0.1,gamma=0.098,tol=0.07)

svrclassifier.fit(X_train,y_train)
X_pred=svrclassifier.predict(X_train)
y_pred=svrclassifier.predict(X_test)


########################
########VALIDATION SCORE#######
########################
# param_range = np.arange(1,20)
# train_scores, test_scores = validation_curve(
#     svrclassifier, X_train, y_train, param_name="C", param_range=param_range,
#     cv=5, scoring="explained_variance", n_jobs=4)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve for SVR")
# plt.xlabel('C')
# plt.ylabel('Explained Variance Score(Y)')
# lw = 2
# plt.ylim(0,1)
# plt.plot(param_range, train_scores_mean, label="Training score",
#             color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Testing score",
#             color="navy", lw=lw)

# plt.legend(loc="best")


# title = 'SVR Learning Curve'
# plot_learning_curve(svrclassifier, title, X_train, y_train, ylim=(0, 1.01),cv=5, n_jobs=4)






print('Training Score is ', metrics.explained_variance_score(y_train,X_pred))
print('Testing Score is ', metrics.explained_variance_score(y_test,y_pred))

print('Training Error is ', metrics.mean_squared_error(y_train,X_pred))
print('Testing Error is ', metrics.mean_squared_error(y_test,y_pred))

#plt.savefig('img/SVR/SVR_Learning_Y.png')
