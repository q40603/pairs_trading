# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:15:35 2020

@author: MAI
"""
import time
import numpy as np
import pandas as pd
import os
from collections import defaultdict

#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table'
tablePath = r'C:\Users\MAI\Desktop\trading_result\after20191128\dilated_cnn_min_open0.85_close0.75'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\fore_lag5_half_thres0.45_open0.85close_0.85'
test_dataPath = r'C:\Users\MAI\Desktop\2016\averageprice'
#test_dataPath = r'C:\Users\MAI\Desktop\half-min\2017\0050'
test_volumePath = r'C:\Users\MAI\Desktop\min-API\accumulate_volume'
tick_trigger_dataPath = r'D:\tick(secs)\2016'
origin_trigger_dataPath = r'C:\Users\MAI\Desktop\2016\minprice'
if half:
    test_csv_name = '_half_min'
else:
    test_csv_name = '_averagePrice_min'


years = ["2016"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
#months = ["01"]
days = ["01","02","03","04","05","06","07","08","09","10",
       "11","12","13","14","15","16","17","18","19","20",
       "21","22","23","24","25","26","27","28","29","30",
       "31"]
def zero():
    return 0
#Clock
start_time = time.time()
count = defaultdict(zero)
totalDay = 0
for year in years:
    for month in months:
        print("Now import: ",month,"-th month")
        for day in days:
            try:
                #Read data
                table = pd.read_csv(os.path.join(tablePath,'{}{}{}_table.csv'.format(year,month,day)))
            except:
                continue
            else:
                totalDay += 1
                stock = [str(table.stock1.iloc[i])+str(table.stock2.iloc[i]) for i in range(len(table))]                    
                for name in stock:
                    count[name] +=1

c = {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}
