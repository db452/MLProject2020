import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_score
from sklearn import metrics
from learningcurve import  plot_learning_curve
from sklearn.metrics.pairwise import euclidean_distances




######Create List of euclidean Distances

def euclidean(a,b):
    dist=[]
    for i in range(len(a)):
        dist.append(np.linalg.norm(a[i]-b[i]))

    c = np.mean(dist)
    return c 
