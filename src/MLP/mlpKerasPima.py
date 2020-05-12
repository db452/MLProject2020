# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import train_test_split

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
train_set_labels = dataset["HasDiabetes"].copy()
train_set = dataset.drop("HasDiabetes", axis=1)

# Rescale input values
scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)

X = train_set_scaled
Y = train_set_labels


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
alpha = 0.001

##############
### MODEL ####
##############

def create_model(act, lr, n, mom, dr, nest, alpha):
    model = Sequential()
    model.add(Dense(n, input_dim=8, activation=act, kernel_regularizer=l2(alpha), name="FirstLayer"))
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
print("Accuracy on TR: ", history.history['acc'][-1]*100)
print("Loss on TR: ", history.history['loss'][-1])
print("Accuracy on TS: ",  history.history['val_acc'][-1]*100)
print("Loss on TS: ",  history.history['val_loss'][-1])

# Plotting Acc & Loss
#plot_results(history)