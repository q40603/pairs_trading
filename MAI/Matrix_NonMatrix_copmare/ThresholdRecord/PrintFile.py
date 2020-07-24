# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:41:44 2020

@author: MAI
"""

import numpy as np
import pandas as pd


count = np.zeros(50)
temp = np.argsort(ConvergeThresOnly2015csv['Normal Close Rate'])[-10:]
count[temp] +=1
temp = np.argsort(ConvergeThresOnly2016csv['Normal Close Rate'])[-10:]
count[temp] +=1
temp = np.argsort(ConvergeThresOnly2017csv['Normal Close Rate'])[-10:]
count[temp] +=1
temp = np.argsort(ConvergeThresOnly2018csv['Normal Close Rate'])[-10:]
count[temp] +=1

print(ConvergeThresOnly2015csv.iloc[1,:])
print(ConvergeThresOnly2016csv.iloc[1,:])
print(ConvergeThresOnly2017csv.iloc[1,:])
print(ConvergeThresOnly2018csv.iloc[1,:])

print(ConvergeThresOnly2015csv.iloc[8,:])
print(ConvergeThresOnly2016csv.iloc[8,:])
print(ConvergeThresOnly2017csv.iloc[8,:])
print(ConvergeThresOnly2018csv.iloc[8,:])

count = np.zeros(50)
temp = np.argsort(ConvergeThresEOnly2015csv['Normal Close Rate'])[-10:]
count[temp] +=1
temp = np.argsort(ConvergeThresEOnly2016csv['Normal Close Rate'])[-10:]
count[temp] +=1
temp = np.argsort(ConvergeThresEOnly2017csv['Normal Close Rate'])[-10:]
count[temp] +=1
temp = np.argsort(ConvergeThresEOnly2018csv['Normal Close Rate'])[-10:]
count[temp] +=1

print(ConvergeThresEOnly2015csv.iloc[1,:])
print(ConvergeThresEOnly2016csv.iloc[1,:])
print(ConvergeThresEOnly2017csv.iloc[1,:])
print(ConvergeThresEOnly2018csv.iloc[1,:])

print(ConvergeThresEOnly2015csv.iloc[8,:])
print(ConvergeThresEOnly2016csv.iloc[8,:])
print(ConvergeThresEOnly2017csv.iloc[8,:])
print(ConvergeThresEOnly2018csv.iloc[8,:])

count = np.zeros(50)
temp = np.argsort(MeanDifThresOnly2015csv['Normal Close Rate'])[-10:]
print(temp)
count[temp] +=1
temp = np.argsort(MeanDifThresOnly2016csv['Normal Close Rate'])[-10:]
print(temp)
count[temp] +=1
temp = np.argsort(MeanDifThresOnly2017csv['Normal Close Rate'])[-10:]
print(temp)
count[temp] +=1
temp = np.argsort(MeanDifThresOnly2018csv['Normal Close Rate'])[-10:]
print(temp)
count[temp] +=1

print(MeanDifThresOnly2015csv.iloc[8,:])
print(MeanDifThresOnly2016csv.iloc[8,:])
print(MeanDifThresOnly2017csv.iloc[8,:])
print(MeanDifThresOnly2018csv.iloc[8,:])
