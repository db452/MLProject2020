from sklearn.preprocessing import MinMaxScaler as Scaler
import pandas as pd
import numpy

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load monk dataset
train_set = pd.read_csv("../dataset/Monk/monks-3-train.csv", header=None, delim_whitespace=True)
test_set = pd.read_csv("../dataset/Monk/monks-3-test.csv", header=None, delim_whitespace=True)

# set columns name
train_set.columns = ["label", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
test_set.columns = ["label", "a1", "a2", "a3", "a4", "a5", "a6", "id"]

# drop id
train_set = train_set.drop("id", axis=1)
test_set = test_set.drop("id", axis=1)

# split attributes from labels
train_set_labels = train_set["label"].copy()
train_set = train_set.drop("label", axis=1)

test_set_labels = test_set["label"].copy()
test_set = test_set.drop("label", axis=1)

# feature scaling
scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)

scaler.fit(test_set)
test_set_scaled = scaler.transform(test_set)

X = train_set_scaled
Y = train_set_labels

X_test = test_set
Y_test = test_set_labels

parameters = {'activation':['tanh','logistic'], 'hidden_layer_sizes':[[4],[6],[8],[16]], 'alpha':[0.003,0.009,0.03,0.09,0.3], 'batch_size':[32, 64, 256], 'learning_rate_init':[0.03,0.1,0.3], 'momentum':[0.7,0.9], 'nesterovs_momentum':[True] }
clf = GridSearchCV(MLPClassifier(verbose=True, solver='sgd', max_iter=1024), parameters, n_jobs=-1, cv=10, scoring='accuracy')
clf.fit(X, Y)
print("/////////")
print(clf.score)
print("/////////")
print("acc: ", clf.score(X_test,Y_test)*100,"%")
print("/////////")
print(clf.best_params_)