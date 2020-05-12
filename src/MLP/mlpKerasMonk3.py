from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler as Scaler

import matplotlib.pyplot as plt
import numpy
import pandas as pd

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

####################
### PLOT ###########
####################

def plot_results(history):
    # summarize and Plot history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.0)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.cla()
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.ylim(0.0, 1.0)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.cla()
    plt.clf()

####################
### LOAD DATASET ###
####################

dataset = pd.read_csv("dataset/Monk/monks-3-train.csv", header=None, delim_whitespace=True)
test_dataset = pd.read_csv("dataset/Monk/monks-3-test.csv", header=None, delim_whitespace=True)


# set columns name
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

#######################
### HYPERPARAMETERS ###
#######################

activation = 'tanh'
learn_rate = 0.2
neurons = 13
dropout = 0.0 # NOT USED
momentum = 0.7
nesterov = True
epochs = 1024
batch_size = 128
alpha= 0.001

##############
### MODEL ####
##############

def create_model(act, lr, n, mom, dr, nest,alpha):
    model = Sequential()
    model.add(Dense(n, input_dim=6, activation=act, kernel_regularizer=l2(alpha), name="FirstLayer"))
    model.add(Dropout(dr, name="FirstDropout"))
    model.add(Dense(1, activation='sigmoid', name="OutputLayer"))
    sgd = optimizers.SGD(lr=lr, momentum=mom, nesterov=nest)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

#####################
### TEST&EVALUATE ###
#####################

# Create & Compile Model
model = create_model(activation, learn_rate, neurons, momentum, dropout, nesterov, alpha)
# Fit the model
history = model.fit(X, Y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy on TR: ", history.history['acc'][-1]*100)
print("Loss on TR: ", history.history['loss'][-1])
print("Accuracy on TS: ", scores[1]*100)
print("Loss on TS: ", scores[0])

# Plotting Acc & Loss
#plot_results(history)