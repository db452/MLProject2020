from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.preprocessing import MinMaxScaler as Scaler

import numpy
import pandas as pd
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

####################
### LOAD DATASET ###
####################

dataset = pd.read_csv("../dataset/MLCup/MLCup-train.csv", header=None)
test_dataset = pd.read_csv("../dataset/MLCup/MLCup-test.csv", header=None)


# set columns name
dataset.columns = ["id", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "label1","label2"]
test_dataset.columns = ["id", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]

######################
### PRE-PROCESSING ###
######################


# drop id
train_set = dataset.drop("id", axis=1)
test_set = test_dataset.drop("id", axis=1)


# split attributes from labels

train_set_labels = train_set.copy()
for i in range(1,11):
    name = "a"+str(i)
    train_set_labels = train_set_labels.drop(name, axis=1)

train_set = train_set.drop("label1", axis=1)
train_set = train_set.drop("label2", axis=1)

# feature scaling
scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)

scaler = Scaler()
scaler.fit(test_set)
test_set_scaled = scaler.transform(test_set)

X = train_set_scaled
Y = train_set_labels

X_test = test_set_scaled

########################
### CROSS-VALIDATION ###
########################

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
tsscores = []
cvloss = []
tsloss = []

#######################
### HYPERPARAMETERS ###
#######################

activation =  'logistic'
learn_rate = 0.2
neurons = 10
dropout = 0.5 # NOT USED
momentum = 0.9
nesterov = True
epochs = 1024
batch_size = 32
decay = 0.00001# NOT USED
alpha= 0.01

##############
### MODEL ####
##############

def create_model(act, lr, n, mom, dec, nest, a, epochs):
    model = MLPRegressor(solver='sgd', activation=act,
                          verbose=True,
                          learning_rate_init=lr,
                          hidden_layer_sizes=(n),
                          momentum=mom,
                          nesterovs_momentum=nest,
                          alpha=a,
                          max_iter=epochs)
    return model


parameters = {'hidden_layer_sizes':[[14],[15],[16]], 'alpha':[0.0, 0.0001, 0.0005], 'batch_size':[16], 'learning_rate_init':[0.1], 'momentum':[0.7] }
clf = GridSearchCV(MLPRegressor(verbose=True, solver='sgd', max_iter=1024), parameters, n_jobs=-1, cv=5, scoring="mean_squared_error")
clf.fit(X, Y)
print("/////////")
print(clf.score)
print("/////////")
#print("acc: ", clf.score(X_test,Y_test)*100,"%")
print("/////////")
print(clf.best_params_)