#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:30:12 2018
@author: pablorr10
"""

import copy as cp
import numpy as np
import pandas as pd
print('Package Imported! ;)')

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
        resp = list(X)[-1]
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
            
        self.names.append(resp)
                
        df = pd.DataFrame(np.zeros(shape=(self.l, (self.p + self.r + 1))), 
                          columns=self.names)
        
        # Populate by rows in the new dataframe
        for i in range(self.l):
            
            slices = np.array([])
            
            # Flatten the lags of predictors
            for p in range(x.shape[1]):
            
                line = X.values[i:self.w + i, p]
                # Reinitialization at every window for ∆T
                if p == 0: line = self.re_init(line)
                    
                # Concatenate the lines in one slice    
                slices = np.concatenate((slices, line)) 
 
            # Incorporate the timestamps where we want to predict
            line = np.array([self.re_init(X.values[i:i+self.w+self.r, 0])[-1]])
            y = np.array(X.values[self.w + i + self.r - 1, -1]).reshape(1,)
            '''
            Just for debugging purpose
            if i == 154:
                a = 2
            '''
            slices = np.concatenate((slices, line, y))

            df.iloc[i,:] = slices
            
        return df

 

##############################################################################
# Reframe Windows to Time Step Method for LSTM
##############################################################################
class WindowToTimesteps():
    
    def __init__(self, data, t):
        '''
        Take the subsets of colums that represents the lags of each varaibles
        and reshape them into the timesteps dimension of a 3D array
        '''
        self.f = int(len(list(data))/t)
        self.data = data
        self. t = t
        
        self.slic = pd.DataFrame(np.zeros(shape=(len(self.data), self.t)))
        self.cake = pd.Panel(np.zeros(shape=(len(self.data), t, self.f)))
        self.cak  = np.zeros(shape=(len(self.data), self.t, self.f)) # Tracking

        
    def collect_timesteps(self):
        # Iterate over the slices       
        for i, j in enumerate(range(0, len(list(self.data)), self.t)):
            
            self.slic.iloc[:,:] = self.data.iloc[:,j:j+self.t].values             
            self.cak[:,:,i] = self.slic.values
            self.cake.iloc[:,:,i] = self.slic.values
      
        return self.cak, self.cake 


##############################################################################
# Reframe Windows to --- for CNN
##############################################################################
class WindowsToPictures():
    
    def __init__(self, window, data):
        '''
        Intuition
        =========
        Take time series data and for every real measured point, create a 
        window of the last x hours for all the parameters, and stack each
        of the windows returning a 3D tensor of [time, features, windows]
        
        The idea is that evey windows will be a 'picture' of the state of
        the process for the last x hours
        
        Parameters
        ==========
        window -> size of the window in time that will represent a picture
        X -> pandas DataFrame with the data to construct the pictures from
        
        Returns
        =======
        3D Tensor of the stacked pictures in the tensors 'window' dimension
        '''
        self.w = window
        self.pic = np.zeros(shape=(self.w, len(list(data))))
        self.prisma = np.zeros(shape=(1, self.pic.shape[0], self.pic.shape[1]))
        self.X = data
        
    def collect_pictures(self):
        j = 0
        # Iterate over the feature where we want to track the real measurements
        for i in range(self.w, len(self.X)):
            # Collect when we have a real measurement T > 0
            if self.X.iloc[i,0] != 0:
                self.pic[:,:] = self.X.iloc[(i - self.w + 1):(i + 1), :].values
                self.prisma = np.concatenate((self.prisma,
                                              self.pic.reshape(1, 
                                                               self.pic.shape[0],
                                                               self.pic.shape[1])), 
                                              axis = 0)
        
        if j == 0: self.prisma = self.prisma[1:,:,:] # Delete zeros from initialization
        j += 1
        
        return self.prisma