from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler as Scaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.regularizers import l2

import numpy
import pandas as pd
import matplotlib.pyplot as plt

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

X_predict = test_set_scaled

########################
### CROSS-VALIDATION ###
########################

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

#######################
### HYPERPARAMETERS ###
#######################

learn_rate = 0.01
neurons = 14
dropout = 0.0
momentum = 0.7
nesterov = True
epochs = 1024
batch_size = 809
decay = 0.00001# NOT USED
alpha= 0.0001


##############
### MODEL ####
##############

def create_model(lr, n, mom, dec, dr, nest,alpha):
    model = Sequential()
    model.add(Dense(n, input_dim=10, kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha), name="FirstLayer"))
    model.add(Dropout(dr, name="FirstDropout"))
    model.add(Dense(2,  name="OutputLayer"))
    sgd = optimizers.SGD(lr=lr,momentum=mom, nesterov=nest)
    #rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=sgd, )
    return model

#####################
### TEST&EVALUATE ###
#####################
# Create & Compile Model
model = create_model(learn_rate, neurons, momentum, decay, dropout, nesterov, alpha)

# Fit the model
history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)
pred = model.predict(X_predict, batch_size=batch_size, verbose=0,)

for i in range(0,pred.size-1):
    print(i+1,",",pred[i][0],",",pred[i][1] )


print("Loss on TR: ", history.history['loss'][-1])

# Plotting Loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.ylim(0.0, 10.0)
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
plt.cla()
plt.clf()

