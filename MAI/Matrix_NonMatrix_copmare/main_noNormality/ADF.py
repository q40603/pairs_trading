# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:11:33 2019

@author: ChaoHsien
"""

from statsmodels.tsa.stattools import adfuller
import numpy as np

class adf():
    
    def __init__(self,data):
        
        self.data = data
    
    def drop_stationary(self):
        
        min_price = np.log(self.data)
        min_price = min_price.dropna(axis = 1)
    
        min_price.index  = np.arange(0,len(min_price),1)
    
        stationary_stock = np.where(min_price.apply(lambda x: adfuller(x)[1] > 0.05 , axis = 0 ) == False )     # 找出定態的股票
        min_price.drop(min_price.columns[stationary_stock], axis=1 , inplace = True)                            # 刪除定態股票，剩餘為有單根序列。
    
        return min_price
        
    def drop_unitroot(self):
        
        min_price = np.log(self.data)
        min_price = min_price.dropna(axis = 1)
    
        min_price.index  = np.arange(0,len(min_price),1)
    
        stationary_stock = np.where(min_price.apply(lambda x: adfuller(x)[1] > 0.05 , axis = 0 ) == True )     # 找出單跟的股票
        min_price.drop(min_price.columns[stationary_stock], axis=1 , inplace = True)                            # 刪除單跟股票，剩餘為定態序列。
    
        return min_price