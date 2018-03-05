#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:21:00 2018
@author: pablorr10
Creation
"""

import time
import copy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from timeseries import WindowSlider
# from sklearn.preprocessing import MinMaxScaler as SCL

np.random.seed(2018)

##############################################################################
# Data Creation
##############################################################################

N = 600

t = np.arange(0, N, 1).reshape(-1,1)
t = np.array([t[i] + np.random.rand(1)/4 for i in range(len(t))])
t = np.array([t[i] - np.random.rand(1)/7 for i in range(len(t))])
t = np.array(np.round(t, 2))

x1 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
x2 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)
x3 = np.round((np.random.random(N) * 5).reshape(-1,1), 2)

n = np.round((np.random.random(N) * 2).reshape(-1,1), 2)

y = np.array([((np.log(np.abs(2 + x1[t])) - x2[t-1]**2) + 0.02*x3[t-3]*np.exp(x1[t-1])) for t in range(len(t))])
y = np.round(y+n, 2)

plt.figure(figsize=(20,6))
plt.plot(t, y, 'b-')
plt.scatter(t, y, color='black', s=10)
plt.show

##############################################################################
# Creation of ∆T
##############################################################################

dataset = pd.DataFrame(np.concatenate((t, x1, x2, x3, y), axis=1), 
                       columns=['t', 'x1', 'x2', 'x3', 'y'])


deltaT = np.array([(dataset.t[i + 1] - dataset.t[i]) for i in range(len(dataset)-1)])
deltaT = np.concatenate((np.array([0]), deltaT))
print('Mean ∆t: %.2f' % np.mean(deltaT))
print('Median ∆t: %.2f' % np.median(deltaT))

dataset.insert(1, '∆t', deltaT)
limit = int(len(dataset)*0.8)

trainset = dataset.iloc[dataset.index < limit]
testset = dataset.iloc[dataset.index >= limit]


##############################################################################
# Creation of the Windows
##############################################################################
w = 5

train_constructor = WindowSlider()
train_windows = train_constructor.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=False)

test_constructor = WindowSlider()
test_windows = test_constructor.collect_windows(testset.iloc[:,1:],
                                                previous_y=False)

train_constructor_y_inc = WindowSlider()
train_windows_y_inc = train_constructor_y_inc.collect_windows(trainset.iloc[:,1:], 
                                                  previous_y=True)

test_constructor_y_inc = WindowSlider()
test_windows_y_inc = test_constructor_y_inc.collect_windows(testset.iloc[:,1:],
                                                previous_y=True)

# Export Windows to Excel
path_to_output = './windows_ts.xlsx'
writer = pd.ExcelWriter(path_to_output, engine='xlsxwriter')

dataset.to_excel(writer, sheet_name='dataset')
trainset.to_excel(writer, sheet_name='trainset')
testset.to_excel(writer, sheet_name='testset')
train_windows.to_excel(writer, sheet_name='train_windows')
test_windows.to_excel(writer, sheet_name='test_windows')
train_windows_y_inc.to_excel(writer, sheet_name='train_windows_y_inc')
test_windows_y_inc.to_excel(writer, sheet_name='test_windows_y_inc')

writer.save()

# Export Windows to HD5
store = pd.HDFStore('./windows_ts.h5')
store['dataset']  = dataset
store['trainset']  = trainset
store['testset']  = testset
store['train_windows']  = train_windows
store['test_windows'] = test_windows
store['train_windows_y_inc']  = train_windows_y_inc
store['test_windows_y_inc'] = test_windows_y_inc
store.close()


'''
# Scaling
window_scaler = SCL()

train_windows = pd.DataFrame(window_scaler.fit_transform(train_windows),
                             columns = train_windows.columns)

test_windows = pd.DataFrame(window_scaler.transform(test_windows),
                            columns = test_windows.columns)
'''
    
##############################################################################
# BASELINE MODELS
##############################################################################

# ________________ Y_pred = current Y ________________ # RMSE = 11.28
bl_trainset = cp.deepcopy(trainset)
bl_testset = cp.deepcopy(testset)

t0 = time.time()
bl_y = pd.DataFrame(bl_testset['y'])
bl_y_pred = bl_y.shift(periods=1)
tF = time.time()

bl_residuals = bl_y_pred - bl_y
bl_rmse = np.sqrt(np.sum(np.power(bl_residuals,2)) / len(bl_residuals))
print('RMSE = %.2f' % bl_rmse)
print('Time to train = %.2f seconds' % (tF - t0))


# Plot Predictions
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
ax2.set_title('Testset - RMSE = %2.f' % bl_rmse)
ax2.legend()

plt.show()    



# ______________ MULTIPLE LINEAR REGRESSION ______________ # RMSE = 8.61
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(trainset.iloc[:,:-1], trainset.iloc[:,-1])

t0 = time.time()
lr_y = testset['y'].values
lr_y_fit = lr_model.predict(trainset.iloc[:,:-1])
lr_y_pred = lr_model.predict(testset.iloc[:,:-1])
tF = time.time()

lr_residuals = lr_y_pred - lr_y
lr_rmse = np.sqrt(np.sum(np.power(lr_residuals,2)) / len(lr_residuals))
print('RMSE = %.2f' % lr_rmse)
print('Time to train = %.2f seconds' % (tF - t0))


# Plot Predictions
f, ax = plt.subplots(1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Baseline Model' , fontsize=20)
plt.title('RMSE = %.2f' % lr_rmse, fontsize = 18)
plt.grid(color='green', linewidth=0.5, alpha=0.5)

plt.scatter(testset.index, lr_y, color='black', s=10)
plt.plot(testset.index, lr_y, color='b', label='Real Test')
plt.plot(testset.index, lr_y_pred, color='r', label='Predicted Test')

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()


# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Baseline Model' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index, trainset['y'], color='b', label='Real Train')
ax1.plot(trainset.index, lr_y_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index, lr_y, color='b', label='Real Test')
ax2.plot(testset.index, lr_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % lr_rmse)
ax2.legend()

plt.show()    



# ___________ MULTIPLE LINEAR REGRESSION ON WINDOWS ___________ # RMSE = 8.61
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(train_windows.iloc[:,:-1], train_windows.iloc[:,-1])

t0 = time.time()
lr_y = test_windows['y'].values
lr_y_fit = lr_model.predict(train_windows.iloc[:,:-1])
lr_y_pred = lr_model.predict(test_windows.iloc[:,:-1])
tF = time.time()

lr_residuals = lr_y_pred - lr_y
lr_rmse = np.sqrt(np.sum(np.power(lr_residuals,2)) / len(lr_residuals))
print('RMSE = %.2f' % lr_rmse)
print('Time to train = %.2f seconds' % (tF - t0))


# Plot Predictions
f, ax = plt.subplots(1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Multiple Linear Regression' , fontsize=20)
plt.title('RMSE = %.2f' % lr_rmse, fontsize = 18)
plt.grid(color='green', linewidth=0.5, alpha=0.5)

plt.scatter(testset.index[w:], lr_y, color='black', s=10)
plt.plot(testset.index[w:], lr_y, color='b', label='Real Test')
plt.plot(testset.index[w:], lr_y_pred, color='r', label='Predicted Test')

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()


# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Multiple Linear Regression' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index[w:], train_windows['y'], color='b', label='Real Train')
ax1.plot(trainset.index[w:], lr_y_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index[w:], lr_y, color='b', label='Real Test')
ax2.plot(testset.index[w:], lr_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % lr_rmse)
ax2.legend()

plt.show()     



# ________ MULTIPLE LINEAR REGRESSION ON WINDOWS Y_INC ________ # RMSE = 8.61
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(train_windows_y_inc.iloc[:,:-1], train_windows_y_inc.iloc[:,-1])

t0 = time.time()
lr_y = test_windows['y'].values
lr_y_fit = lr_model.predict(train_windows_y_inc.iloc[:,:-1])
lr_y_pred = lr_model.predict(test_windows_y_inc.iloc[:,:-1])
tF = time.time()

lr_residuals = lr_y_pred - lr_y
lr_rmse = np.sqrt(np.sum(np.power(lr_residuals,2)) / len(lr_residuals))
print('RMSE = %.2f' % lr_rmse)
print('Time to train = %.2f seconds' % (tF - t0))


# Plot Predictions
f, ax = plt.subplots(1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - MLR with previous t values' , fontsize=20)
plt.title('RMSE = %.2f' % lr_rmse, fontsize = 18)
plt.grid(color='green', linewidth=0.5, alpha=0.5)

plt.scatter(testset.index[w:], lr_y, color='black', s=10)
plt.plot(testset.index[w:], lr_y, color='b', label='Real Test')
plt.plot(testset.index[w:], lr_y_pred, color='r', label='Predicted Test')

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.xlabel('Time')
plt.ylabel('Response')
plt.legend()
plt.show()


# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - MLR with previous y values' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index[w:], train_windows['y'], color='b', label='Real Train')
ax1.plot(trainset.index[w:], lr_y_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index[w:], lr_y, color='b', label='Real Test')
ax2.plot(testset.index[w:], lr_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % lr_rmse)
ax2.legend()

plt.show()       
