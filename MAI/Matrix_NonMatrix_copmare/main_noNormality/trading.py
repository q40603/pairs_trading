#!/usr/bin/env python3
#- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:52:28 2018

@author: chaohsien
"""

#import plotlib.pyplot as plt
import accelerate_formation
import accelerate_trading
import ADF
import time
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA, ARIMA

# one-day formation period-----------------------------------------------------

if __name__ == '__main__':
    
    start = time.time()
    
#    warnings.filterwarnings('ignore')
    
    # --------------------------------------------------------------------------
    # 計算大盤每日波動,使用 ariam model predict 隔日的波動
    
    '''
    data = pd.read_csv("^TWII.csv")

    data = data.dropna(how='any',axis = 0)

    data.index  = np.arange(0,len(data),1)

    volatility = []
    for i in range(len(data)):

        vol = (data['High'][i] - data['Low'][i])/data['Low'][i]

        if data['Open'][i] < data['Close'][i]: 
    
            volatility.append(vol)
        
        else:
        
            volatility.append(-vol)

    volatility = pd.DataFrame(volatility) ; volatility.columns = ['vol']

    volatility = pd.concat([data['Date'],volatility],axis=1)
    '''
    #---------------------------------------------------------------------------
    
    model = 0
    
    year = ["2018"]
    
    month = ["01","02","03","04","05","06","07","08",
             "09","10","11","12"]
#    month = ["01","02","03"]
#    month = ["04","05","06"]
#    month = ["07","08","09"]
#    month = ["10","11","12"]
#    month = ["01"]
    
    day = ["01","02","03","04","05","06","07","08","09","10",
           "11","12","13","14","15","16","17","18","19","20",
           "21","22","23","24","25","26","27","28","29","30",
           "31"]
#    day = ["22"]
    
    strategy_return = []
    prof = []
    profit = []
    trade_day = []
    break1_pos = 0 ; break2_pos = 0 ; break3_pos = 0 ; break4_pos = 0 ; break5_pos = 0 ;
    break1_neg = 0 ; break2_neg = 0 ; break3_neg = 0 ; break4_neg = 0 ; break5_neg = 0 ;
    loss1 = 0 ; loss2 = 0 ; loss3 = 0 ; loss4 = 0 ; loss5 = 0 ; 
    lossend1_pos = 0 ; lossend2_pos = 0 ; lossend3_pos = 0 ; lossend4_pos = 0 ; lossend5_pos = 0 ; 
    lossend1_neg = 0 ; lossend2_neg = 0 ; lossend3_neg = 0 ; lossend4_neg = 0 ; lossend5_neg = 0 ; 
    normal1 = 0 ; normal2 = 0 ; normal3 = 0 ; normal4 = 0 ; normal5 = 0 ; 
    num1 = 0 ; num2 = 0 ; num3 = 0 ; num4 = 0 ; num5 = 0 ; num6 = 0
    for m in range(len(month)):
        
        for i in range(len(day)):
        
            date = ''.join( [ year[0] , month[m] , day[i] ] ) ; Date = year[0] + "/" + str(int(month[m])) + "/" + str(int(day[i]))
            
            # 讀取資料 --------------------------------------------------------------------------------------------------------------------------------
            try:
            
                # 讀取台股資料(day1)
                day1_0050 = pd.read_csv( ''.join([ "C:/Users/MAI/Desktop/", year[0] ,"/averageprice/" , date , "_averagePrice_min.csv"]) , encoding="utf-8")
                                
            
                # ----------------------------------------------------------------------------------------------------------------------------------------
                # predicit 大盤波動
            
                volatility_threshold = 0.0001
            
                interval = 244                                    # 用前幾筆資料modeling
                
                # 程式防呆,所以寫在try裡
                #endpoint = int(np.array(np.where(volatility['Date']==Date)) - 1)
            
            except:
        
                continue
            '''
            startpoint = int(endpoint - interval)
            
            da = volatility['vol'][startpoint:endpoint] ; da.index = np.arange(0,len(da),1)
            
            order = sm.tsa.arma_order_select_ic(da,ic='bic',max_ar=3,max_ma=3)['bic_min_order'] 

            model = ARMA(da,order=order)
            
            try:
                
                results = model.fit()
                
            except:
                
                # 序列不平穩，做一次差分,並重新找order(p,q)
                diff1 = np.diff(da)
            
                order = sm.tsa.arma_order_select_ic(diff1,ic='bic',max_ar=3,max_ma=3)['bic_min_order']
            
                order = list(order) ; order.insert(1,1) ; order = tuple(order)
                model = ARIMA(da,order=order)
        
                results = model.fit()
            
            predicit = float(model.predict(params=results.params,start=len(da),end=len(da) ))
            '''
            # 1:波動大,0:波動小--------------------------------------------------------------------------------------------------
            
            predicit = 0.5
            
            if abs(predicit) >= volatility_threshold:
                
                flag = 1                # 大盤波動較大
            
            else:
            
                flag = 0                #  大盤波動過小,加入trend stationary 交易策略
                
            #-----------------------------------------------------------------------------------------------------------------------------------------
            # data 整理
            #day1 = pd.concat([day1_0050,day1_0051] , axis = 1)
            
            day1 = day1_0050
            
            day1.index = np.arange(0,len(day1),1)
            day1 = day1.drop(index=np.arange(0,16,1)) ; day1.index = np.arange(0,len(day1),1)

            # formation period and trading period -----------------------------------------------------------------------------------------------------------
            
            print(date)
            
            # 一天只有532半分鐘
            formate_time = 150       # 建模時間長度
            trade_time = 100         # 回測時間長度
            
            for j in range(1):    # 一天建模??次
                
                day1_1 = day1.iloc[(trade_time * j) : (formate_time + (trade_time * j)),:] 
                
                day1_1.index  = np.arange(0,len(day1_1),1)
#                print('is adf?')
                unitroot_stock = ADF.adf.drop_stationary(ADF.adf(day1_1))
                
#                print('hi')
                
                a = accelerate_formation.pairs_trading(unitroot_stock,flag)
                
                try:
                
                    table = accelerate_formation.pairs_trading.formation_period( a )                                 # 共整合配對表格
                
                except:
                    
                    print("table error")
                    
                    prof.append([0,0,0,0,0,0,0,0,0,0])
                    
                    trade_day.append([date])
                
                    continue
            
                path = r"C:/Users/MAI/Desktop/trading_result/after20200204/trading_period_matrix_ver1.0/table/20200609NoNormality/"
                
                table.to_csv( ''.join([ path , date ,"_table.csv" ]) , index = False )                  #寫入方式選擇wb，否則有空行               

    end = time.time()
    
    elapsed = ( end - start ) / 3600
    
    print("time taken",elapsed,"hours")
    
    