from sklearn.neural_network import MLPClassifier
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

dataset = pd.read_csv("dataset/Monk/monks-3-train.csv", header=None, delim_whitespace=True)
test_dataset = pd.read_csv("dataset/Monk/monks-3-test.csv", header=None, delim_whitespace=True)


dataset.columns = ["label", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
test_dataset.columns = ["label", "a1", "a2", "a3", "a4", "a5", "a6", "id"]

######################
### PRE-PROCESSING ###
######################


# drop id
train_set = dataset.drop("id", axis=1)
test_set = test_dataset.drop("id", axis=1)


# split attributes from labels
train_set_labels = train_set["label"].copy()
train_set = train_set.drop("label", axis=1)

test_set_labels = test_set["label"].copy()
test_set = test_set.drop("label", axis=1)

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
Y_test = test_set_labels

########################
### CROSS-VALIDATION ###
########################

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
tsscores = []
cvloss = []

#######################
### HYPERPARAMETERS ###
#######################

activation = 'tanh'
learn_rate = 0.1
neurons = 8
dropout = 0.5 # NOT USED
momentum = 0.7
nesterov = True
epochs = 1024
batch_size = 32
decay = 0.00001# NOT USED
alpha= 0.1

##############
### MODEL ####
##############

def create_model(act, lr, n, mom, dec, nest, a, epochs):
    model = MLPClassifier(solver='sgd', activation=act,
                          learning_rate_init=lr,
                          hidden_layer_sizes=(n),
                          momentum=mom,
                          nesterovs_momentum=nest,
                          alpha=a,
                          max_iter=epochs)
    return model

#####################
### TEST&EVALUATE ###
#####################

# Create & Compile Model
model = create_model(activation, learn_rate, neurons, momentum, decay, nesterov, alpha, epochs)

for train, test in kfold.split(X, Y):
    model.fit(X[train], Y[train])
    cvloss.append(model.loss_)
    scores = model.score(X[test], Y[test])
    # print("Processing... acc: %.2f%%" % (scores * 100))
    print("Processing...")
    tsscores.append(model.score(X_test, Y_test) * 100)
    cvscores.append(scores * 100)

    # Fit the model
    model.fit(X[train], Y[train])
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print("Accuracy on TR: %.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
print("Loss on TR: ", numpy.mean(cvloss), " (+/- ", numpy.std(cvloss), ")")
print("//////////")
print("Accuracy on TS: %.2f%% (+/- %.2f%%)" % (numpy.mean(tsscores), numpy.std(tsscores)))

title = "Learning Curves"
plot_learning_curve(model, title, X, Y, [0.0, 1.1],10)
plt.show()
