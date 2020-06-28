import pandas as pd  
import numpy as np  
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV


train = pd.read_csv("src/dataset/MLCup/Internal Testing/train-self.csv")


dbdata = train


X_train = dbdata.iloc[:, np.arange(20)]
y_train = dbdata.iloc[:, 21]

##Performs the girdsearch
svregressor = SVR(coef0=0,kernel='rbf',shrinking=True,C=8,tol=0.003,epsilon=0.29,gamma=0.23)
parameters = {
}


clf = RandomizedSearchCV(svregressor, parameters,scoring='neg_mean_squared_error', iid=False,cv=5,verbose=10,n_jobs=-1)
clf.fit(X_train,y_train)  


print(clf,file=open('src/GridSearch/Results/SVRoutput.log', 'a'))
print(clf.best_estimator_,file=open('src/GridSearch/Results/SVRoutput.log', 'a'))
print(clf.best_score_,file=open('src/GridSearch/Results/SVRoutput.log', 'a'))
print(clf.best_params_,file=open('src/GridSearch/Results/SVRoutput.log', 'a'))