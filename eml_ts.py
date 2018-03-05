#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:13:37 2018
@author: pablorr10
EML Algorithm to train an NN with a single hidden layer
"""

import time
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.preprocessing import MinMaxScaler as SCL

np.random.seed(2018)


##############################################################################
# IMPORT DATA AND WINDOWS
##############################################################################
path_to_data = './windows_ts.h5'

dataset = pd.read_hdf(path_to_data, 'dataset')
eml_trainset = pd.read_hdf(path_to_data, 'trainset')
eml_testset = pd.read_hdf(path_to_data, 'testset')

train_windows = pd.read_hdf(path_to_data, 'train_windows')
test_windows = pd.read_hdf(path_to_data, 'test_windows')
train_windows_eml_inc = pd.read_hdf(path_to_data, 'train_windows_y_inc')
test_windows_eml_inc = pd.read_hdf(path_to_data, 'test_windows_y_inc')

# Bring here the window used
w = 5

##############################################################################
# CREATION OF THE NETWORK
##############################################################################        
class Network(object):

    def __init__(self, input_dim, hidden_dim=10, output_dim=1):
        '''
        Neural Network object 
        '''
        self.N = input_dim
        self.M = hidden_dim
        self.O = output_dim
        
        self.W1 = np.matrix(np.random.rand(self.N, self.M))
        self.W2 = np.matrix(np.random.rand(self.M, self.O))
        
        self.U = 0
        self.V = 0
        self.S = 0
        self.H = 0
        self.alpha = 0 # for regularization
        
    # Helper function
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-0.1 * x)) - 0.5
    
    def predict(self, x):
        '''
        Forward pass to calculate the ouput
        '''
        x = np.matrix(x)
        y = self.sigmoid(x @ self.W1) @ self.W2
        return y
    
    def train(self, x, y):
        '''
        Compute W2 that lead to minimal LS
        '''
        X = np.matrix(x)
        Y = np.matrix(y)
        self.H = np.matrix(self.sigmoid(X @ self.W1))
        H = cp.deepcopy(self.H)
        
        self.svd(H)
        iH = np.matrix(self.V) @ np.matrix(np.diag(self.S)).I @ np.matrix(self.U).T

        self.W2 = iH * Y
        print('W2 values updated..')
        #return np.linalg.norm(H @ self.W2 - Y)
        return H @ self.W2 - Y
    
    def svd(self, h):
        '''
        Compute the Singular Value Decomposition of a matrix H
        '''
        H = np.matrix(h)
        self.U, self.S, Vt = np.linalg.svd(H, full_matrices=False)
        self.V = np.matrix(Vt).T
        print('SVD computed.. calculating Pseudoinverse..')
        return np.matrix(self.U), np.matrix(self.S), np.matrix(self.V)


    def update():
        '''
        Include new samples - Increment SVD
        '''
        pass
    
    def downdate():
        '''
        Remove old samples - Decrement SVD
        '''
        pass


##############################################################################
# TRAIN THE NETWORK AND PREDICT - Without previous y
############################################################################## 
in_dim = train_windows.shape[1] - 1
NN = Network(input_dim=in_dim, hidden_dim=20, output_dim=1)
t0 = time.time()
eml_residuals = NN.train(x = train_windows.iloc[:,:-1], 
                     y = train_windows.iloc[:,-1].values.reshape(-1,1))
tF = time.time()

# U,S,V = NN.svd(NN.H)


fit = NN.predict(train_windows.iloc[:,:-1])
predictions = NN.predict(test_windows.iloc[:,:-1])

'''
eml_fit = scaler.inverse_transform(fit)
eml_pred = scaler.inverse_transform(predictions)
'''

eml_fit = cp.deepcopy(fit)
eml_pred = cp.deepcopy(predictions)

eml_residuals = eml_pred - eml_testset.iloc[w:, -1].values.reshape(-1,1)
eml_rmse = np.sqrt(np.sum(np.power(eml_residuals,2)) / len(eml_residuals))
print('RMSE = %.2f' % eml_rmse)
print('Time to train %.2f' % (tF - t0))


##############################################################################
# TRAIN THE NETWORK AND PREDICT - With previous y
############################################################################## 
in_dim = train_windows_eml_inc.shape[1] - 1
NN = Network(input_dim=in_dim, hidden_dim=20, output_dim=1)
t0 = time.time()
eml_residuals = NN.train(x = train_windows_eml_inc.iloc[:,:-1], 
                     y = train_windows_eml_inc.iloc[:,-1].values.reshape(-1,1))
tF = time.time()

# U,S,V = NN.svd(NN.H)


fit_inc = NN.predict(train_windows_eml_inc.iloc[:,:-1])
predictions_inc = NN.predict(test_windows_eml_inc.iloc[:,:-1])

'''
eml_fit = scaler.inverse_transform(fit)
eml_pred = scaler.inverse_transform(predictions)
'''

eml_fit_inc = cp.deepcopy(fit_inc)
eml_pred_inc = cp.deepcopy(predictions_inc)

eml_residuals_inc = eml_pred_inc - eml_testset.iloc[w:, -1].values.reshape(-1,1)
eml_rmse_inc = np.sqrt(np.sum(np.power(eml_residuals_inc,2)) / len(eml_residuals_inc))
print('RMSE = %.2f' % eml_rmse_inc)
print('Time to train %.2f' % (tF - t0))



# Plot Predictions
f, ax = plt.subplots(1, figsize=(20,12))
plt.suptitle('Actual vs Predicted - Extreme Machine Learning' , fontsize=20)
plt.title('RMSE without previous Y = %.2f \n RMSE with previous Y = %.2f' 
          % (eml_rmse, eml_rmse_inc), fontsize = 18)
plt.grid(color='green', linewidth=0.5, alpha=0.5)

plt.scatter(eml_testset.index, eml_testset.y, color='black', s=10)
plt.plot(eml_testset.index, eml_testset.y, color='b', label='Real Test')
plt.plot(eml_testset.index[w:], eml_pred, color='r', label='Predicted Test')
plt.plot(eml_testset.index[w:], eml_pred_inc, 
         color='m', linewidth=0.4, label='Predicted Test prev Y included')

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.xlabel('Time')
plt.ylabel('eml_response')
plt.legend()
plt.show()


# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Extreme Machine Learning' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(eml_trainset.index, eml_trainset['y'], color='b', label='Real Train')
ax1.plot(eml_trainset.index[w:], eml_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(eml_testset.index, eml_testset.y, color='b', label='Real Test')
ax2.plot(eml_testset.index[w:], eml_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % eml_rmse)
ax2.legend()
plt.show()    


##############################################################################
# Importance of timestamps
############################################################################## 
from sklearn.model_selection import train_test_split

trainset, _ = train_test_split(eml_trainset, test_size=0.6, shuffle=True)
trainset = trainset.sort_index()
testset, _ = train_test_split(eml_testset, test_size=0.6, shuffle=True)
testset = testset.sort_index()

train = trainset.reset_index()
test = testset.reset_index()

plt.figure(figsize=(20,6))
plt.scatter(eml_trainset.index, eml_trainset['y'], color='black', s=10)
plt.scatter(eml_testset.index, eml_testset['y'], color='black', s=10)
plt.scatter(trainset.index, trainset['y'], color='green', s=20)
plt.scatter(testset.index, testset['y'], color='green', s=20)
plt.plot(eml_trainset['y'], 'b-')
plt.plot(eml_testset['y'], 'r-')
plt.plot(trainset['y'], 'g--')
plt.plot(testset['y'], 'g--')
plt.show

deltaT = np.array([(train.t[i + 1] - train.t[i]) for i in range(len(trainset)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))
print('Mean ∆t: %.2f' % np.mean(deltaT))
print('Median ∆t: %.2f' % np.median(deltaT))

from timeseries import WindowSlider

train_constructor = WindowSlider()
train_windows_with_t = train_constructor.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=False)

test_constructor = WindowSlider()
test_windows_with_t = test_constructor.collect_windows(testset.iloc[:,1:],
                                                previous_y=False)


# 1. Keeping Timestamp
in_dim = train_windows_with_t.shape[1] - 1
NN = Network(input_dim=in_dim, hidden_dim=20, output_dim=1)
eml_residuals_with_t = NN.train(x = train_windows_with_t.iloc[:,:-1], 
                     y = train_windows_with_t.iloc[:,-1].values.reshape(-1,1))

eml_fit_with_t = NN.predict(train_windows_with_t.iloc[:,:-1])
eml_pred_with_t = NN.predict(test_windows_with_t.iloc[:,:-1])

eml_residuals_with_t = eml_pred_with_t - test_windows_with_t.iloc[:, -1].values.reshape(-1,1)
eml_rmse_with_t = np.sqrt(np.sum(np.power(eml_residuals_with_t,2)) / len(eml_residuals_with_t))
print('RMSE = %.2f' % eml_rmse_with_t)
print('Time to train %.2f' % (tF - t0))


# 2. Removing Timestamp
train_windows_no_t = train_windows_with_t.drop(train_windows_with_t.columns[0:5], axis=1)
train_windows_no_t.drop(train_windows_with_t.columns[-2], axis=1, inplace=True)

test_windows_no_t = test_windows_with_t.drop(test_windows_with_t.columns[0:5], axis=1)
test_windows_no_t.drop(test_windows_with_t.columns[-2], axis=1, inplace=True)

in_dim = train_windows_no_t.shape[1] - 1
NN = Network(input_dim=in_dim, hidden_dim=20, output_dim=1)
eml_residuals_no_t = NN.train(x = train_windows_no_t.iloc[:,:-1], 
                     y = train_windows_no_t.iloc[:,-1].values.reshape(-1,1))

eml_fit_no_t = NN.predict(train_windows_no_t.iloc[:,:-1])
eml_pred_no_t = NN.predict(test_windows_no_t.iloc[:,:-1])

eml_residuals_no_t = eml_pred_no_t - test_windows_no_t.iloc[:, -1].values.reshape(-1,1)
eml_rmse_no_t = np.sqrt(np.sum(np.power(eml_residuals_no_t,2)) / len(eml_residuals_no_t))
print('RMSE = %.2f' % eml_rmse_no_t)
print('Time to train %.2f' % (tF - t0))



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,20))
plt.suptitle('Accuracy difference giving or not the timestamps')
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.t, trainset.y, color='b', label='Real Train')
ax1.plot(eml_fit_with_t.t, trainset.y, color='b', label='Real Train')

# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Extreme Machine Learning' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(eml_trainset.index, eml_trainset['y'], color='b', label='Real Train')
ax1.plot(eml_trainset.index[w:], eml_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(eml_testset.index, eml_testset.y, color='b', label='Real Test')
ax2.plot(eml_testset.index[w:], eml_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % eml_rmse)
ax2.legend()
plt.show()    