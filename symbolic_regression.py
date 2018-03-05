#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:56:58 2018
@author: pablorr10
Symbolic Regression
"""

import time
import copy as cp
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from sklearn.preprocessing import MinMaxScaler as SCL

##############################################################################
# IMPORT DATA AND WINDOWS
##############################################################################
path_to_data = './windows_ts.h5'

dataset = pd.read_hdf(path_to_data, 'dataset')
sr_trainset = pd.read_hdf(path_to_data, 'trainset')
sr_testset = pd.read_hdf(path_to_data, 'testset')

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
function_set = ['add', 'sub', 'mul', 'div', 'sin', 'log'] #, xexp]
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
sr_residuals = sr_y_pred - sr_testset.iloc[5:,-1].values.reshape(-1,1)
sr_rmse = np.sqrt(np.sum(np.power(sr_residuals,2)) / len(sr_residuals))
print('RMSE = %f' % sr_rmse)
print('Time to train %.2f' % (tF - t0))



##############################################################################
# TRAIN THE NETWORK AND PREDICT - With previous y
############################################################################## 
# Train
t0 = time.time()
model.fit(train_windows_sr_inc.values[:,:-1], train_windows_sr_inc.values[:,-1])
tF = time.time()

# Predict
sr_y_fit_inc = model.predict(train_windows_sr_inc.values[:,:-1]).reshape(-1,1)
sr_y_pred_inc = model.predict(test_windows_sr_inc.values[:,:-1]).reshape(-1,1)

# Calculating Errors
sr_residuals_inc = sr_y_pred_inc - sr_testset.iloc[5:,-1].values.reshape(-1,1)
sr_rmse_inc = np.sqrt(np.sum(np.power(sr_residuals_inc,2)) / len(sr_residuals_inc))
print('RMSE = %f' % sr_rmse_inc)
print('Time to train %.2f' % (tF - t0))



# Plot Predictions
def plot_sr():
    f, ax = plt.subplots(1, figsize=(20,12))
    plt.suptitle('Actual vs Predicted - Symbolic Regression' , fontsize=20)
    plt.title('RMSE without previous Y = %.2f \n RMSE with previous Y = %.2f' 
              % (sr_rmse, sr_rmse_inc), fontsize = 18)
    plt.grid(color='green', linewidth=0.5, alpha=0.5)
    
    plt.scatter(sr_testset.index, sr_testset.y, color='black', s=10)
    plt.plot(sr_testset.index, sr_testset.y, color='b', label='Real Test')
    plt.plot(sr_testset.index[w:], sr_y_pred, color='r', label='Predicted Test')
    plt.plot(sr_testset.index[w:], sr_y_pred_inc, 
             color='k', linewidth=0.4, label='Predicted Test prev Y included')
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.xlabel('Time')
    plt.ylabel('sr_response')
    plt.legend()
    plt.show()

plot_sr()

# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,8))
plt.suptitle('Actual vs Predicted - Symbolic Regression , fontsize=20')
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(sr_trainset.index, sr_trainset['y'], color='b', label='Real Train')
ax1.plot(sr_trainset.index[w:], sr_y_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(sr_testset.index, sr_testset.y, color='b', label='Real Test')
ax2.plot(sr_testset.index[w:], sr_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % sr_rmse)
ax2.legend()
plt.show()    

'''
##############################################################################
# Another Library
import array, random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create('FitnessMax', base.Fitness, wights=(1.0,))
creator.create('Individual', array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register(')
'''