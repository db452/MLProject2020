from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold,learning_curve,validation_curve
from sklearn.preprocessing import MinMaxScaler as Scaler

import matplotlib.pyplot as plt
import numpy
import pandas as pd

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

dataset = pd.read_csv("dataset/Pima/pima-indians-diabetes.csv", header=None)

# Because thr CSV doesn't contain any header, we add column names
# using the description from the original dataset website
dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"]

######################
### PRE-PROCESSING ###
######################

# Calculate the median value for BMI and substitute it  where values are 0
median_bmi = dataset['BMI'].median()
dataset['BMI'] = dataset['BMI'].replace(to_replace=0, value=median_bmi)

# Calculate the median value for BloodP and substitute it  where values are 0
median_bloodp = dataset['BloodP'].median()
dataset['BloodP'] = dataset['BloodP'].replace(to_replace=0, value=median_bloodp)

# Calculate the median value for PlGlcConc and substitute it  where values are 0
median_plglcconc = dataset['PlGlcConc'].median()
dataset['PlGlcConc'] = dataset['PlGlcConc'].replace(to_replace=0, value=median_plglcconc)

# Calculate the median value for SkinThick and substitute it  where values are 0
median_skinthick = dataset['SkinThick'].median()
dataset['SkinThick'] = dataset['SkinThick'].replace(to_replace=0, value=median_skinthick)

# Calculate the median value for TwoHourSerIns and substitute it  where values are 0
median_twohourserins = dataset['TwoHourSerIns'].median()
dataset['TwoHourSerIns'] = dataset['TwoHourSerIns'].replace(to_replace=0, value=median_twohourserins)

# Split into input (X) and target (Y)
Y = dataset["HasDiabetes"].copy()
X = dataset.drop("HasDiabetes", axis=1)

# Rescale input values
scaler = Scaler()
scaler.fit(X)
X = scaler.transform(X)

########################
### CROSS-VALIDATION ###
########################

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

#######################
### HYPERPARAMETERS ###
#######################

activation =  'tanh'
learn_rate = 0.15
neurons = 16
dropout = 0.0
momentum = 0.7
nesterov = True
epochs = 512
batch_size = 307
decay = 0.00001
alpha = 0.001
##############
### MODEL ####
##############

def create_model(act, lr, n, mom, dec, dr, nest, a,  epochs):
    model = MLPClassifier(activation=act, solver='sgd', hidden_layer_sizes=(n), learning_rate_init=lr, alpha=a, max_iter=epochs, nesterovs_momentum=nest, momentum=mom)
    return model

#####################
### TEST&EVALUATE ###
#####################
model = create_model(activation, learn_rate, neurons, momentum, decay, dropout, nesterov, alpha, epochs)

for train, test in kfold.split(X,Y):
    # Create & Compile Model
    model.fit(X[train], Y[train])
    scores = model.score(X[test], Y[test])
    print("Processing... acc: %.2f%%" % (scores * 100))
    cvscores.append(scores * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
title = "Learning Curves"
plot_learning_curve(model, title, X, Y, [0.0, 1.1],10)
plt.show()