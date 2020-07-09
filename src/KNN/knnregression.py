import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn import metrics
from learningcurve import  plot_learning_curve
from euclidean_distances import euclidean

train = pd.read_csv("src\\dataset\\MLCup\\Model Testing\\train-self.csv",header=None)
test = pd.read_csv("src\\dataset\\MLCup\\Model Testing\\test-self.csv",header=None)  

dbdata = train
dbdata2 = test

    

X_train_X = dbdata.iloc[:, np.arange(20)]
y_train_X = dbdata.iloc[:, 20]
X_test_X = dbdata2.iloc[:, np.arange(20)]
y_test_X = dbdata2.iloc[:, 20]

X_train_Y = dbdata.iloc[:, np.arange(20)]
y_train_Y = dbdata.iloc[:, 21]
X_test_Y = dbdata2.iloc[:, np.arange(20)]
y_test_Y = dbdata2.iloc[:, 21]



knn = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=15)
knn2 = KNeighborsRegressor(algorithm='auto', metric='manhattan', weights='distance',p=2,n_neighbors=7)



knn.fit(X_train_X,y_train_X)
X_pred_X=knn.predict(X_train_X)
y_pred_X=knn.predict(X_test_X)

knn2.fit(X_train_Y,y_train_Y)
X_pred_Y=knn2.predict(X_train_Y)
y_pred_Y=knn2.predict(X_test_Y)



# ########################
# ########GRAPH#######
# ########################
# param_range = np.arange(1,30)
# train_scores, test_scores = validation_curve(
#     knn, X_train, y_train, param_name="n_neighbors", param_range=param_range,
#     cv=5, scoring="explained_variance", n_jobs=4)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve for KNR")
# plt.xlabel('Number of Nearest Neighbors')
# plt.ylabel('Explained Variance Score(Y)')
# lw = 2
# plt.ylim(0.95,1)
# plt.plot(param_range, train_scores_mean, label="Training score",
#             color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Testing score",
#             color="navy", lw=lw)

# # plt.legend(loc="best")
##########################Learning Curve
# title = 'KNR Learning Curve'
# plot_learning_curve(knn, title, X_train, y_train, ylim=(0, 1.01),cv=5, n_jobs=4)


###################### X, Y of train (and pred)

train_xy = np.column_stack((y_train_X,y_train_Y))
trainpred_xy = np.column_stack((X_pred_X,X_pred_Y))


###################### X, Y of test (and pred)


test_xy = np.column_stack((y_test_X,y_test_Y))
pred_xy = np.column_stack((y_pred_X,y_pred_Y))

################## X,Y of test Pred





print('Training Score is ', euclidean(train_xy,trainpred_xy))
print('Testing Score is ', euclidean(test_xy,pred_xy))


#plt.savefig('img/KNR/KNR_learning_curve_Y.png')
