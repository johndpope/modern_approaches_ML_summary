#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:56:01 2018
@author: pablorr10
"""

import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from timeseries import WindowSlider, WindowToTimesteps

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from beautifultable import BeautifulTable as BT
from sklearn.preprocessing import MinMaxScaler as SCL

# Versions check
print('Numpy version:', np.__version__)
print('Matplotlib version:', mpl.__version__)
print('Tensorflow Version:', tf.__version__)

'''
######################## Helper Functions ################################
def window_to_timesteps(data, t):
    
    # Basic units of the algorithm
    f = int(len(list(data))/t)
    slic = pd.DataFrame(np.zeros(shape=(len(data), t)))
    cake = pd.Panel(np.zeros(shape=(len(data), t, f)))
    cak  = np.zeros(shape=(len(data), t, f)) # Tracking

    # Iterate over the slices   
    step = cp.copy(t)
    for i, j in enumerate(range(0, len(list(data)), step)):
        
        slic.iloc[:,:] = data.iloc[:,j:j+step].values             
        cak[:,:,i] = slic.values
        cake.iloc[:,:,i] = slic.values
  
    return cak, cake 
    
######################## Helper Functions ################################    
'''


##############################################################################
# IMPORT DATA AND WINDOWS
##############################################################################
path_to_data = './windows_ts.h5'

# Note this time we have to scale before constructing the windows
dataset = pd.read_hdf(path_to_data, 'dataset')
limit = int(len(dataset)*0.8)
trainset = dataset.iloc[dataset.index < limit]
testset = dataset.iloc[dataset.index >= limit]

# Reference the response for future unscaling
yscaler = SCL()
yscaler.fit(trainset['y'].values.reshape(-1,1))

scaler = SCL()
lstm_trainset = pd.DataFrame(scaler.fit_transform(trainset), columns = dataset.columns)
lstm_testset = pd.DataFrame(scaler.transform(testset), columns = dataset.columns)

'''
# If you want to see that we have not change the data ;)
plt.figure(figsize=(20,6))
plt.plot(testset.t, testset.y, 'b-')
plt.scatter(testset.t, testset.y, color='black', s=10)
plt.show
'''

train_constructor = WindowSlider()
lstm_train_windows = train_constructor.collect_windows(lstm_trainset.iloc[:,1:], 
                                                       previous_y=False)

test_constructor = WindowSlider()
lstm_test_windows = test_constructor.collect_windows(lstm_testset.iloc[:,1:],
                                                     previous_y=False)

train_constructor_y_inc = WindowSlider()
lstm_train_windows_y_inc = train_constructor_y_inc.collect_windows(lstm_trainset.iloc[:,1:], 
                                                                   previous_y=True)

test_constructor_y_inc = WindowSlider()
lstm_test_windows_y_inc = test_constructor_y_inc.collect_windows(lstm_testset.iloc[:,1:],
                                                                 previous_y=True)

# Bring here the window used - for the plots
w = 5
timesteps = w
window_to_ts = WindowToTimesteps(lstm_train_windows.iloc[:,:-2], t=timesteps)
predictors, _ = window_to_ts.collect_timesteps()


###############################################################################
# 2 - SPLIT MATRIX OF FEATURES AND RESPONSE AND RESHAPING TO LSTM WINDOW METHOD
# - Without previous y
###############################################################################
trainX = lstm_train_windows.iloc[:, :-2].values
trainX = trainX.reshape(lstm_train_windows.shape[0], timesteps, trainset.shape[1]-2)

trainY = lstm_train_windows.iloc[:, -1].values

testX  = lstm_test_windows.iloc[:, :-2].values
testX = testX.reshape(testX.shape[0], timesteps, testset.shape[1]-2)

testY  = lstm_test_windows.iloc[:, -1].values
 
table = BT(max_width=3000)
table.column_headers = ['Set', 'Samples', 'Timesteps', 'Features']
table.append_row(['trainX', trainX.shape[0], trainX.shape[1], trainX.shape[2]])
table.append_row(['testX',  testX.shape[0],  testX.shape[1],  testX.shape[2]])
table.append_row(['trainY', trainY.shape[0],     1,           1])
table.append_row(['TestY',  testY.shape[0],      1,           1])
print(table)


##############################################################################
# CREATION AND TRAINING OF THE NETWORK - Without previous y
##############################################################################        
if 'model' in locals(): del model
if 'history' in locals(): del history

# 1 - CREATE LSTM
model = Sequential()
model.add(LSTM(1, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()


# 1.2 - TRAIN LSTM
t0 = time.time()
history = model.fit(trainX, trainY, epochs=80, batch_size=16, 
                    validation_data=(testX, testY), verbose=1, shuffle=True)
tF = time.time()

# 1.3. - Plot history of training
plt.figure(figsize=(20,6))
plt.title('Evolution of the Error in Train and Test sets during training', 
          fontsize=20)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


##############################################################################
# PREDICTING AND CALCULATING ERRORS - Without previous y
lstm_fit = model.predict(trainX)
lstm_pred = model.predict(testX)

# Invert scaling for forecast
lstm_y_fit = yscaler.inverse_transform(lstm_fit)
lstm_y_pred = yscaler.inverse_transform(lstm_pred)

# Calculate Errors
lstm_residuals = testset['y'].values - lstm_y_pred
lstm_rmse = np.sqrt(np.sum(np.power(lstm_residuals,2))/ len(testset))
print('Test RMSE: %.3f' % lstm_rmse)
print('Time to train %.2f' % (tF - t0))


# 2 - CREATE STACKED LSTM WITH NEXT TIMESTEP
# 2.1 - Create Data
trainX = np.concatenate((lstm_train_windows.iloc[:,-2].values.reshape(-1,1), lstm_fit), axis=1)
trainY = lstm_train_windows.iloc[:,-1].values.reshape(-1,1)

testX = np.concatenate((lstm_test_windows.iloc[:,-2].values.reshape(-1,1), lstm_pred), axis=1)
testY = lstm_test_windows.iloc[:,-1].values.reshape(-1,1)

# 2.2 CREATE DNN
if 'model2' in locals(): del model2
if 'history2' in locals(): del history2

model2 = Sequential()
model2.add(Dense(32, input_shape=(2, )))
model2.add(Dropout(0.3))
model2.add(Dense(32))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer='adam')
model2.summary()

# 2.3 - TRAIN DNN
t0 = time.time()
history2 = model2.fit(trainX, trainY, epochs=40, batch_size=16, 
                    validation_data=(testX, testY), verbose=1, shuffle=True)
tF = time.time()

# 2.3. - Plot history of training
plt.figure(figsize=(20,6))
plt.title('Evolution of the Error in Train and Test sets during training', 
          fontsize=20)
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.legend()
plt.show()


# Plot Predictions
def plot_lstm():
    f, ax = plt.subplots(1, figsize=(20,12))
    plt.suptitle('Actual vs Predicted - LSTM' , fontsize=20)
    plt.title('RMSE without previous Y = %.2f' % (lstm_rmse), fontsize = 18)
    plt.grid(color='green', linewidth=0.5, alpha=0.5)
    
    plt.scatter(testset.index, testset.y, color='black', s=10)
    plt.plot(testset.index, testset.y, color='b', label='Real Test')
    plt.plot(testset.index[w:], lstm_y_pred, color='r', label='Predicted Test')
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.xlabel('Time')
    plt.ylabel('lstm_response')
    plt.legend()
    plt.show()

plot_lstm()

# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Extreme Machine Learning' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index, trainset['y'], color='b', label='Real Train')
ax1.plot(trainset.index[w:], lstm_y_fit, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index, testset.y, color='b', label='Real Test')
ax2.plot(testset.index[w:], lstm_y_pred, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % lstm_rmse)
ax2.legend()
plt.show()    




###############################################################################
# 2 - SPLIT MATRIX OF FEATURES AND RESPONSE AND RESHAPING TO LSTM WINDOW METHOD
# - Without previous y
###############################################################################
trainX = lstm_train_windows_y_inc.iloc[:, :-1].values
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])

trainY = lstm_train_windows_y_inc.iloc[:, -1].values

testX  = lstm_test_windows_y_inc.iloc[:, :-1].values
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

testY  = lstm_test_windows_y_inc.iloc[:, -1].values
 
table = BT(max_width=3000)
table.column_headers = ['Set', 'Samples', 'Timesteps', 'Features']
table.append_row(['trainX', trainX.shape[0], trainX.shape[1], trainX.shape[2]])
table.append_row(['testX',  testX.shape[0],  testX.shape[1],  testX.shape[2]])
table.append_row(['trainY', trainY.shape[0],     1,           1])
table.append_row(['TestY',  testY.shape[0],      1,           1])
print(table)

##############################################################################
# CREATION AND TRAINING OF THE NETWORK - With previous y
##############################################################################        
if 'model' in locals(): del model
if 'history' in locals(): del history

# 1 - CREATE LSTM
model_inc = Sequential()
model_inc.add(LSTM(150, input_shape=(1, trainX.shape[2])))
model_inc.add(Dense(1))
model_inc.compile(loss='mse', optimizer='adam')
model_inc.summary()


# 2 - TRAIN LSTM
t0 = time.time()
history_inc = model_inc.fit(trainX, trainY, epochs=30, batch_size=128, 
                    validation_data=(testX, testY), verbose=1, shuffle=True)
tF = time.time()

# 2.1. - Plot history of training
plt.figure(figsize=(20,6))
plt.title('Evolution of the Error in Train and Test sets during training', 
          fontsize=20)
plt.plot(history_inc.history['loss'], label='train')
plt.plot(history_inc.history['val_loss'], label='test')
plt.legend()
plt.show()

##############################################################################
# PREDICTING AND CALCULATING ERRORS - Without previous y
############################################################################## 
lstm_fit_inc = model_inc.predict(trainX)
lstm_pred_inc = model_inc.predict(testX)

# Invert scaling for forecast
lstm_y_fit_inc = yscaler.inverse_transform(lstm_fit_inc)
lstm_y_pred_inc = yscaler.inverse_transform(lstm_pred_inc)

# Calculate Errors
lstm_residuals_inc = testset['y'].values - lstm_y_pred_inc
lstm_rmse_inc = np.sqrt(np.sum(np.power(lstm_residuals_inc,2))/ len(testset))
print('Test RMSE: %.3f' % lstm_rmse_inc)
print('Time to train %.2f' % (tF - t0))


# Plot Predictions
def plot_lstm_inc():
    f, ax = plt.subplots(1, figsize=(20,12))
    plt.suptitle('Actual vs Predicted - LSTM' , fontsize=20)
    plt.title('RMSE without previous Y = %.2f' % (lstm_rmse), fontsize = 18)
    plt.grid(color='green', linewidth=0.5, alpha=0.5)
    
    plt.scatter(testset.index, testset.y, color='black', s=10)
    plt.plot(testset.index, testset.y, color='b', label='Real Test')
    plt.plot(testset.index[w:], lstm_y_pred_inc, color='r', label='Predicted Test')
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.xlabel('Time')
    plt.ylabel('lstm_response')
    plt.legend()
    plt.show()

plot_lstm_inc()

# Complete Plot
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,6))
plt.suptitle('Actual vs Predicted - Extreme Machine Learning' , fontsize=20)
ax1.grid(color='green', linewidth=0.5, alpha=0.5)
ax2.grid(color='green', linewidth=0.5, alpha=0.5)

ax1.plot(trainset.index, trainset['y'], color='b', label='Real Train')
ax1.plot(trainset.index[w:], lstm_y_fit_inc, color='r', label='Predicted Train')
ax1.legend()
ax1.set_title('Trainset')

ax2.plot(testset.index, testset.y, color='b', label='Real Test')
ax2.plot(testset.index[w:], lstm_y_pred_inc, color='r', label='Predicted Test')
ax2.set_title('Testset - RMSE = %2.f' % lstm_rmse)
ax2.legend()
plt.show()    



