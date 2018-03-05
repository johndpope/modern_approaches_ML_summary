#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:56:58 2018
@author: pablorr10
Symbolic Regression
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from sklearn.preprocessing import MinMaxScaler as SCL

##############################################################################
# IMPORT DATA AND WINDOWS
##############################################################################
path_to_data = './windows_ts.h5'

dataset = pd.read_hdf(path_to_data, 'dataset')
trainset = pd.read_hdf(path_to_data, 'trainset')
testset = pd.read_hdf(path_to_data, 'testset')

train_windows = pd.read_hdf(path_to_data, 'train_windows')
test_windows = pd.read_hdf(path_to_data, 'test_windows')
train_windows_sr_inc = pd.read_hdf(path_to_data, 'train_windows_y_inc')
test_windows_sr_inc = pd.read_hdf(path_to_data, 'test_windows_y_inc')

# Bring here the window used
w = 5


##############################################################################
# CREATION OF THE MODEL
##############################################################################        
import gplearn as gpl        
from gplearn.genetic import SymbolicRegressor

def _xexp( x ):
    a = np.exp(x); 
    a[ np.abs(a) > 1e+9 ] = 1e+9
    return a    

xexp = gpl.functions.make_function( function = _xexp, name='xexp', arity=1 )
#function_set = ['add', 'sub', 'mul', 'div', 'sin', 'log'] #, xexp]
function_set = ['add', 'sub', 'mul', 'div']

if 'model' in locals(): del model
model = SymbolicRegressor(population_size = 1000, tournament_size=5,
                          generations = 25, stopping_criteria=0.00001,
                          function_set = function_set, metric='rmse',
                          p_crossover=0.65, p_subtree_mutation=0.15,
                          p_hoist_mutation=0.05, p_point_mutation=0.1,
                          verbose = 1, random_state = None) #, n_jobs = -1)


##############################################################################
# TRAIN THE NETWORK AND PREDICT - Without previous y
############################################################################## 
# Train
t0 = time.time()
model.fit(train_windows.values[:,:-1], train_windows.values[:,-1])
tF = time.time()

# Predict
sr_y_fit = model.predict(train_windows.values[:,:-1]).reshape(-1,1)
sr_y_pred = model.predict(test_windows.values[:,:-1]).reshape(-1,1)

# Calculating Errors
sr_residuals = sr_y_pred - testset.iloc[5:,-1].values.reshape(-1,1)
sr_rmse = np.sqrt(np.sum(np.power(sr_residuals,2)) / len(sr_residuals))
print('RMSE = %f' % sr_rmse)
print('Time to train %.2f' % (tF - t0))
print(model._program)


# Plot predictions
f, ax = plt.subplots(1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - MLR with previous t values' , fontsize=20)
plt.title('RMSE = %.2f' % sr_rmse, fontsize = 18)
plt.grid(color='green', linewidth=0.5, alpha=0.5)

plt.scatter(testset.index, testset.y, color='black', s=10)
plt.plot(testset.index, testset.y, color='b', label='Real Test')
plt.plot(testset.index[w:], sr_y_pred, color='r', label='Predicted Test')

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()


# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Gaussian Process' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index[w:], train_windows['y'], color='b', label='Real Train')
ax1.plot(trainset.index[w:], sr_y_fit, color='r',
         linewidth=0.8, alpha=0.8, label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index[w:], test_windows['y'], color='b', label='Real Test')
ax2.plot(testset.index[w:], sr_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % sr_rmse)
ax2.legend()

plt.show()     




##############################################################################
# TRAIN THE NETWORK AND PREDICT - With previous y
############################################################################## 
if 'model_inc' in locals(): del model_inc
model_inc = SymbolicRegressor(population_size = 1000, tournament_size=5,
                          generations = 25, stopping_criteria=0.00001,
                          function_set = function_set, metric='rmse',
                          p_crossover=0.65, p_subtree_mutation=0.15,
                          p_hoist_mutation=0.05, p_point_mutation=0.1,
                          verbose = 1, random_state = None) #, n_jobs = -1)
# Train
t0 = time.time()
model_inc.fit(train_windows_sr_inc.values[:,:-1], train_windows_sr_inc.values[:,-1])
tF = time.time()

# Predict
sr_y_fit_inc = model_inc.predict(train_windows_sr_inc.values[:,:-1]).reshape(-1,1)
sr_y_pred_inc = model_inc.predict(test_windows_sr_inc.values[:,:-1]).reshape(-1,1)

# Calculating Errors
sr_residuals_inc = sr_y_pred_inc - testset.iloc[5:,-1].values.reshape(-1,1)
sr_rmse_inc = np.sqrt(np.sum(np.power(sr_residuals_inc,2)) / len(sr_residuals_inc))
print(model._program)
print('RMSE = %f' % sr_rmse_inc)
print('Time to train %.2f' % (tF - t0))


# Plot predictions
f, ax = plt.subplots(1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - MLR with previous t values' , fontsize=20)
plt.title('RMSE = %.2f' % sr_rmse_inc, fontsize = 18)
plt.grid(color='green', linewidth=0.5, alpha=0.5)

plt.scatter(testset.index, testset.y, color='black', s=10)
plt.plot(testset.index, testset.y, color='b', label='Real Test')
plt.plot(testset.index[w:], sr_y_pred_inc, color='r', label='Predicted Test')

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()


# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Gaussian Process' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index[w:], train_windows['y'], color='b', label='Real Train')
ax1.plot(trainset.index[w:], sr_y_fit_inc, color='r',
         linewidth=0.8, alpha=0.8, label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index[w:], test_windows['y'], color='b', label='Real Test')
ax2.plot(testset.index[w:], sr_y_pred_inc, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % sr_rmse_inc)
ax2.legend()

plt.show()     

