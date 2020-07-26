#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 18:52:28 2018

@author: chaohsien
"""

from formation_period import formation_period_single #, formation_period_pair
import accelerate_formation
import accelerate_trading
import ADF
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# one-day formation period-----------------------------------------------------

if __name__ == '__main__':
    
    year = ["2015"]
    
    month = ["06"]
    
    day = ["01","02","03","04","05","06","07","08","09","10",
           "11","12","13","14","15","16","17","18","19","20",
           "21","22","23","24","25","26","27","28","29","30",
           "31"]
    
    prof = []
    trade_day = []
    for i in range(len(day)-1):
        
        date = ''.join( [ year[0] , month[0] , day[i] ] ) 
        
        # 讀取資料 --------------------------------------------------------------------------------------------------------------------------------
        try:
            
            # 讀取台股資料(day1)
            day1 = pd.read_csv( ''.join(["averageprice/" , date , "_averagePrice_min.csv"]) , encoding="utf-8").drop([266,267,268,269,270])
            day1_tick = pd.read_csv( ''.join(["minprice/" , date , "_min_stock.csv"]) , encoding="utf-8").drop([266,267,268,269,270])
        except:
            print("error")
            continue
        
        finally:
            
            name = day1.columns
            day2 = pd.DataFrame(columns=name)
            
            for k in range(1,len(day)-i):
                
                date1 = ''.join( [ year[0] , month[0] , day[i+k] ] )
            
                try:
                    
                    # 讀取台股資料(day2)
                    day2 = pd.read_csv( ''.join(["averageprice/" , date1 , "_averagePrice_min.csv"]) , encoding="utf-8").drop([266,267,268,269,270])
                    day2_tick = pd.read_csv( ''.join(["minprice/" , date1 , "_min_stock.csv"]) , encoding="utf-8").drop([266,267,268,269,270])
                    
                    break
                
                except:
            
                    continue
                
        if day2.size == 0:
            
            break
            
    # formation period and trading period -----------------------------------------------------------------------------------------------------------
    
        print(date)
    
        day1 = pd.concat([day1,day2] , join='inner' , ignore_index=True )                     
        day1_tick = pd.concat([day1_tick,day2_tick] , join='inner' , ignore_index=True)
        
        # 一天只有266分鐘
        formate_time = 150      # 建模時間長度
        trade_time = 115        # 回測時間長度
        
        for j in range(1):    # 一天建模??次
            
            day1_1 = day1.iloc[(trade_time * j) : (formate_time + (trade_time * j) - 1),:]
            
            #day1_1 = pd.concat([day1,day2.iloc[0:formate_time,:]])
            day1_1.index  = np.arange(0,len(day1_1),1)
            print(day1_1) 
            unitroot_stock = ADF.adf.drop_stationary(ADF.adf(day1_1))
                
            a = accelerate_formation.pairs_trading(unitroot_stock)
            
            try:
                
                table = accelerate_formation.pairs_trading.formation_period( a )                                 # 共整合配對表格
                print(table)
            except:
                
                prof.append([0,0,0,0])
                
                continue
                
            #table_single = formation_period_single( ADF.adf.drop_unitroot(ADF.adf(day1_1)) )                     # 單支股票為定態序列表格
        
            # one-day trading period-------------------------------------------------------

            day1_2 = day1.iloc[ (formate_time + trade_time * j) : (formate_time + trade_time * (j+1) ) , : ] ; day1_2.index  = np.arange(0,len(day1_2),1) 
            
            day1_2.index  = np.arange(0,len(day1_2),1)
            print(day1_2)    
            day1_tick_2 = day1_tick.iloc[ (formate_time + (trade_time * j) - 1) : (formate_time + trade_time * (j+1) ) , : ]
            
            day1_tick_2.index  = np.arange(0,len(day1_tick_2),1)
            print(day1_tick_2)  


            print(day1)
            capital = 3000           # 每組配對資金300萬
            maxi = 5                 # 股票最大持有張數
            k1 = 1.5                 # 開倉門檻倍數
            k2 = 10                  # 停損門檻倍數
            
            # pair -----------------------------------------------------------------------------------------------
            
            m1 = accelerate_trading.trading(table , formate_time , day1_2 , day1_tick_2 , k1 , k2 , day1 , maxi , 0 , capital)

            m2 = accelerate_trading.trading(table , formate_time , day1_2 , day1_tick_2 , k1 , k2 , day1 , maxi , 0.003 , capital)
            
            result  = accelerate_trading.trading.backtest_table(m1)                                                                                                                                                                                              
            result1 = accelerate_trading.trading.backtest_table(m2)
            
            mix = pd.merge( result , result1 , on=["stock1","stock2"] , how="outer")
                
            table = pd.merge( table , mix , on=['stock1','stock2'] , how='outer')
            
            #result = pd.DataFrame(result) ; result.columns = ["profit without cost","open number","return"]
            #result1 = pd.DataFrame(result1) ; result1.columns = ["profit","open number","return"]
            
            table = pd.concat([table,result,result1],axis=1)
            
            # single ---------------------------------------------------------------------------------------------
                
            #m3 = accelerate_trading.trading(table_single , day1_2 , day1_tick_2 , k1 , 3 , day1 , maxi , 0 )
            #m4 = accelerate_trading.trading(table_single , day1_2 , day1_tick_2 , k1 , 3 , day1 , maxi , 0.003 )
                
            #result2  = accelerate_trading.trading.single( m3 )              
            #result3 = accelerate_trading.trading.single( m4 )
                
            #table_single = pd.concat([table_single,result2,result3],axis=1)
            
            #path = "C:/Users/ChaoHsien/Desktop/trading_result/"
                
            #table.to_csv( ''.join([ path , date , "_table.csv" ]) , index = False )                  #寫入方式選擇wb，否則有空行
            #table_single.to_csv( ''.join([ path , date[i] , "_table_single.csv" ]) , index = False )
                
            prof.append( [ sum(table.iloc[:,10]) , sum(table.iloc[:,11]) , sum(table.iloc[:,13]), sum(table.iloc[:,14]) ])#, 
                          #sum(table_single.iloc[:,4]) , sum(table_single.iloc[:,5]), sum(table_single.iloc[:,7]), sum(table_single.iloc[:,8])] )
            
            trade_day.append([date,date1])
        break
            
    prof = np.array(prof)
    #trade_day = np.array(trade_day)
    # print(prof)      
    print( [sum(prof[:,0]) , sum(prof[:,1]) , sum(prof[:,2]) , sum(prof[:,3]) ] )
            
    #ppp = np.hstack((trade_day,prof))
            
    







