#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:21:00 2018
@author: pablorr10
"""

from math import sqrt, exp
import time
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler as SCL

np.random.seed(2018)

# Data Preparation
t = np.arange(0, 200, 1).reshape(-1,1)
t = np.array([t[i] + np.random.rand(1)/4 for i in range(len(t))])
t = np.array([t[i] - np.random.rand(1)/7 for i in range(len(t))])
t = np.array(np.round(t, 2))

x1 = np.round((np.random.random(200) * 5).reshape(-1,1), 2)
x2 = np.round((np.random.random(200) * 5).reshape(-1,1), 2)
x3 = np.round((np.random.random(200) * 5).reshape(-1,1), 2)
y = np.array([((x1[t] - x2[t-1]**2) + 0.02*x3[t-3]*exp(x1[t-1])) for t in range(len(t))])
y = np.round(y, 2)
'''
plt.figure(figsize=(20,6))
plt.plot(t, y, 'b-')
plt.scatter(t, y, color='black', s=10)
plt.show
'''
# Data Aggregation and create ∆T
dataset = pd.DataFrame(np.concatenate((t, x1, x2, x3, y), axis=1), 
                       columns=['t', 'x1', 'x2', 'x3', 'y'])


deltaT = np.array([(dataset.t[i + 1] - dataset.t[i]) for i in range(len(dataset)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))
print('Mean ∆T: %.2f' % np.mean(deltaT))
print('Median ∆T: %.2f' % np.median(deltaT))

dataset.insert(1, '∆T', deltaT)
limit = int(len(dataset)*0.8)

trainset = dataset.iloc[dataset.index < limit]
testset = dataset.iloc[dataset.index >= limit]

##############################################################################
# BASELINE MODELS
##############################################################################

# ________________ Y_pred = current Y ________________ # RMSE = 

bl_trainset = cp.deepcopy(trainset)
bl_testset = cp.deepcopy(testset)

bl_y = pd.DataFrame(bl_testset['y'])
bl_y_pred = bl_y.shift(periods=1)


bl_residuals = bl_y_pred - bl_y
bl_rmse = np.sqrt(np.sum(np.power(bl_residuals,2)) / len(bl_residuals))
print('RMSE = %f' % bl_rmse)


# Plot Predictions
def plot_bl():
    f, ax = plt.subplots(1, figsize=(20,6))
    plt.suptitle('Actual vs Predicted - Baseline Model' , fontsize=20)
    plt.title('RMSE = %.2f' % bl_rmse, fontsize = 18)
    plt.grid(color='green', linewidth=0.5, alpha=0.5)
    
    plt.scatter(bl_testset.index, bl_y, color='black', s=10)
    plt.plot(bl_testset.index, bl_y, color='b', label='Real Test')
    plt.plot(bl_testset.index, bl_y_pred, color='r', label='Predicted Test')
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.show()

plot_bl()

# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Baseline Model' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(bl_trainset.index, bl_trainset['y'], color='b', label='Real Train')
ax1.plot(bl_trainset.index, bl_trainset['y'].shift(1), color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(bl_testset.index, bl_y, color='b', label='Real Test')
ax2.plot(bl_testset.index, bl_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset')
ax2.legend()

plt.show()    


##############################################################################
# Creation of the Windows
##############################################################################
class WindowSlider(object):
    
    def __init__(self, window_size = 5):        
        '''
        Window Slider object
        ====================
        w: window_size - number of time steps to look back
        o: offset between last reading and temperature
        r: response_size - number of time steps to predict
        l: maximum length to slide - (#observation - w)
        p: final predictors - (#predictors * w)
        '''
        self.w = window_size
        self.o = 0
        self.r = 1       
        self.l = 0
        self.p = 0
        self.names = []
        
    def re_init(self, arr):
        '''
        Helper function to initializate to 0 a vector
        '''
        arr = np.cumsum(arr)
        return arr - arr[0]
                

    def collect_windows(self, X, window_size=5, offset=0, previous_y=False):
        '''
        Input: X is the input matrix, each column is a variable
        Returns: diferent mappings window-output
        '''
        cols = len(list(X)) - 1
        N = len(X)
        
        self.o = offset
        self.w = window_size
        self.l = N - (self.w + self.r) + 1
        if not previous_y: self.p = cols * (self.w)
        if previous_y: self.p = (cols + 1) * (self.w)
        
        # Create the names of the variables in the window
        # Check first if we need to create that for the response itself
        if previous_y: x = cp.deepcopy(X)
        if not previous_y: x = X.drop(X.columns[-1], axis=1)  
        
        for j, col in enumerate(list(x)):        
                
            for i in range(self.w):
                
                name = col + ('(%d)' % (i+1))
                self.names.append(name)
        
        # Incorporate the timestamps where we want to predict
        for k in range(self.r):
            
            name = '∆t' + ('(%d)' % (self.w + k + 1))
            self.names.append(name)
            
        self.names.append('Y')
                
        df = pd.DataFrame(np.zeros(shape=(self.l, (self.p + self.r + 1))), 
                          columns=self.names)
        
        # Populate by rows in the new dataframe
        for i in range(self.l - 1):
            
            slices = np.array([])
            
            # Flatten the lags of predictors
            for p in range(x.shape[1]):
            
                line = X.values[i:self.w + i, p]
                # Reinitialization at every window for ∆T
                if p == 0: line = self.re_init(line)
                    
                # Concatenate the lines in one slice    
                slices = np.concatenate((slices, line)) 
 
            # Incorporate the timestamps where we want to predict
            line = self.re_init(X.values[i:i+self.w+self.r+1, 0])[len(line)-self.r+1:-1]
            y = np.array(X.values[self.w + i + self.r, -1]).reshape(1,)
            slices = np.concatenate((slices, line, y))
            
            df.iloc[i,:] = slices
            
        return df.iloc[:-self.r,:]

# Construct Windows
train_constructor = WindowSlider()
train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=False)

test_constructor = WindowSlider()
test_windows = test_constructor.collect_windows(testset.iloc[:,1:],
                                                previous_y=False)


'''
# Scaling
window_scaler = SCL()

train_windows = pd.DataFrame(window_scaler.fit_transform(train_windows),
                             columns = train_windows.columns)

test_windows = pd.DataFrame(window_scaler.transform(test_windows),
                            columns = test_windows.columns)

'''
    
