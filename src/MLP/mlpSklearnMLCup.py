from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold,learning_curve,validation_curve
from sklearn.preprocessing import MinMaxScaler as Scaler

import numpy
import pandas as pd
import matplotlib.pyplot as plt

##############
### PLOT #####
##############

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=8, n_jobs=None, train_sizes=numpy.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = numpy.mean(train_scores, axis=1)
    train_scores_std = numpy.std(train_scores, axis=1)
    test_scores_mean = numpy.mean(test_scores, axis=1)
    test_scores_std = numpy.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

####################
### LOAD DATASET ###
####################

dataset = pd.read_csv("dataset/MLCup/MLCup-train.csv", header=None)
test_dataset = pd.read_csv("dataset/MLCup/MLCup-test.csv", header=None)


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

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
tsscores = []
cvloss = []
tsloss = []

#######################
### HYPERPARAMETERS ###
#######################

activation =  'logistic'
learn_rate = 0.1
neurons = 14
dropout = 0.0 # NOT USED
momentum = 0.7
nesterov = True
epochs = 4096
batch_size = 16
alpha= 0.0001

##############
### MODEL ####
##############

def create_model(act, lr, n, mom, nest, a, epochs):
    model = MLPRegressor(solver='sgd', activation=act,
                          verbose=True,
                          learning_rate_init=lr,
                          hidden_layer_sizes=(n),
                          momentum=mom,
                          nesterovs_momentum=nest,
                          alpha=a,
                          max_iter=epochs,
                          tol=0.0000001)
    return model

#####################
### TEST&EVALUATE ###
#####################
model = create_model(activation, learn_rate, neurons, momentum, nesterov, alpha, epochs)
# Fit the model
model.fit(X, Y)
print("Model Loss on TR: ", model.loss_)



