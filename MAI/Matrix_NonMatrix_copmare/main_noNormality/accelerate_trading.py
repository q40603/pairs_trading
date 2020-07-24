# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:08:05 2018

@author: chuchu0936
"""

#from formation_period import formation_period
#import matplotlib.pyplot as plt
#from statsmodels.tsa.stattools import adfuller
from multiprocessing import Process,Manager,Pool
#from trading_period_with_timetrend import pairs
#from trading_period_with_timetrend import pairs
from cost import tax , slip
#from integer import num_weight
#from MTSA import fore_chow
import pandas as pd
import numpy as np

class trading(object):
    
    def __init__( self , table , formate_time , min_data , tick_data , open_time , close_time , stop_loss_time , day1 , maxi , tax_cost , cost_gate , capital , model_CNN ,flag ):
        
        self.table          =  table                      # formation period table
        self.formate_time   =  formate_time               # 建模長度
        self.min_data       =  min_data                   # 回測期的加權平均股價
        self.tick_data      =  tick_data                  # 回測期的開倉平倉股價
        self.open_time      =  open_time                  # 開倉倍數
        self.close_time     =  close_time                 # 平倉倍數
        self.stop_loss_time =  stop_loss_time             # 停損倍數
        self.day1           =  day1                       # 建模期+回測期的加權平均股價
        self.maxi           =  maxi                       # 最大股票持有張數
        self.tax_cost       =  tax_cost                   # 交易成本
        self.cost_gate      =  cost_gate                  # 交易門檻
        self.capital        =  capital                    # 每組配對最大資金上限
        self.model_CNN      =  model_CNN
        self.flag           =  flag
        self.stock1_name    =  []
        self.stock2_name    =  []
        self.profit         =  []
        self.open_num       =  []
        self.rt             =  []
        self.std            =  []
        self.skew           =  []
        self.timetrend      =  []
        self.pos            =  []
        
    # 單支股票進行回測 -------------------------------------------------------------------------------
    def single(self):
    
        min_price = self.day1
        min_price = min_price.dropna(axis = 1)
        min_price.index  = np.arange(0,len(min_price),1)
        
        num = np.arange(0,len(self.table),1)
        
        open_num = []
        profit = []
        rt = []
        for pair in num:
            
            if 1.5 * self.table.stdev[pair] < self.tax_cost :
                
                open_num.append(0)
                profit.append(0)
                rt.append(0)
                
                continue
                
            #pair = 55
            spread = np.log(self.min_data[ self.table.stock[pair] ])
        
            #x = np.arange(0,61)
            #plt.plot(spread)
            #plt.axhline(y=close,color='r')
            #plt.axhline(y=up_open,color='r')
            #plt.axhline(y=down_open,color='r')
            #plt.axhline(y=close+stop_loss,color='green')
            #plt.axhline(y=close-stop_loss,color='green')
        
            up_open = self.table.mu[pair] + self.table.stdev[pair] * self.open_time                      # 上開倉門檻
            down_open = self.table.mu[pair] - self.table.stdev[pair] * self.open_time                    # 下開倉門檻
            
            stop_loss = self.table.stdev[pair] * self.stop_loss_time                                     # 停損門檻
            
            close = self.table.mu[pair]                                                                  # 平倉(均值)
            
            M = round( self.table.zcr[pair] * len(spread) )                                              # 平均持有時間
            
            trade = 0                                                                                    # 計算開倉次數
            spread_return = 0
            
            position = 0                                                                             # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2:強制平倉
            pos = []
            stock_profit = []
        
            for i in range( len(spread)-2 ):
        
                if position == 0 and len(spread)-i > M :                                                 # 之前無開倉
        
                    if ( spread[i] - up_open ) * ( spread[i+1] - up_open ) < 0 :
                    
                        position = -1
                    
                        stock_payoff = slip( self.tick_data[self.table.stock[pair]][(i+2)] , 1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost ) # 交易成本
                    
                        spreadprice = stock_payoff                                                       # 計算報酬用
                    
                        trade = trade + 1
                
                    elif ( spread[i] - down_open ) * ( spread[i+1] - down_open ) < 0 :
                    
                        position = 1
                    
                        stock_payoff = -slip( self.tick_data[self.table.stock[pair]][(i+2)] , -1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost ) # 交易成本
                    
                        spreadprice = stock_payoff                                                       # 計算報酬用
                    
                        trade = trade + 1
                
                    else: 
            
                        position = 0
        
                        stock_payoff = 0
            
                elif position == -1:                                                                     # 之前有開空倉，平空倉
        
                    if ( spread[i] - close ) * ( spread[i+1] - close ) < 0 :
        
                        position = 0                                                                     # 平倉
            
                        stock_payoff = -slip( self.tick_data[self.table.stock[pair]][(i+2)] , -1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost ) # 交易成本
                    
                        spread_return = spread_return + ( spreadprice + stock_payoff)/abs(spreadprice)   # 每次交易報酬做累加(最後除以交易次數做平均)
                
                    elif ( spread[i] - (close + stop_loss) ) * ( spread[i+1] - (close + stop_loss) ) < 0 :
                
                        position = -2                                                                    # 碰到停損門檻，強制平倉
            
                        stock_payoff = -slip( self.tick_data[self.table.stock[pair]][(i+2)] , -1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost ) # 交易成本
                    
                        spread_return = spread_return + ( spreadprice + stock_payoff)/abs(spreadprice)   # 每次交易報酬做累加(最後除以交易次數做平均)
                    
                    elif i == (len(spread)-3) :                                                          # 回測結束，強制平倉
            
                        position = 0
            
                        stock_payoff = -slip( self.tick_data[self.table.stock[pair]][len(self.tick_data)-1] , -1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost ) # 交易成本
                    
                        spread_return = spread_return + ( spreadprice + stock_payoff)/abs(spreadprice)   # 每次交易報酬做累加(最後除以交易次數做平均)
                    
                    else: 
            
                        position = -1
        
                        stock_payoff = 0
            
                elif position == 1:                                                                       # 之前有開多倉，平多倉
        
                    if ( spread[i] - close ) * ( spread[i+1] - close ) < 0 :
        
                        position = 0                                                                      # 平倉
            
                        stock_payoff = slip( self.tick_data[self.table.stock[pair]][(i+2)] , 1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost )  # 交易成本
                    
                        spread_return = spread_return + ( spreadprice + stock_payoff)/abs(spreadprice)    # 每次交易報酬做累加(最後除以交易次數做平均)
                    
                    elif ( spread[i] - (close - stop_loss) ) * ( spread[i+1] - (close - stop_loss) ) < 0 :
                
                        position = -2                                                                     # 碰到停損門檻，強制平倉
            
                        stock_payoff = slip( self.tick_data[self.table.stock[pair]][(i+2)] , 1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost )  # 交易成本
                    
                        spread_return = spread_return + ( spreadprice + stock_payoff)/abs(spreadprice)    # 每次交易報酬做累加(最後除以交易次數做平均)
                    
                    elif i == (len(spread)-3) :                                                           # 回測結束，強制平倉
            
                        position = 0
            
                        stock_payoff = slip( self.tick_data[self.table.stock[pair]][len(self.tick_data)-1] , 1 )
                        
                        stock_payoff , stock2_payoff = tax(stock_payoff , 1 , position , self.tax_cost )  # 交易成本
                    
                        spread_return = spread_return + ( spreadprice + stock_payoff)/abs(spreadprice)    # 每次交易報酬做累加(最後除以交易次數做平均)
                    
                    else: 
            
                        position = 1
        
                        stock_payoff = 0
                else:
                
                    if position == -2:
                    
                        position = -2
                    
                        stock_payoff = 0
                        
                    else:
                    
                        position = 0                                                                       # 剩下時間少於預期開倉時間，則不開倉，避免損失
        
                        stock_payoff = 0
            
                pos.append(position)
        
                stock_profit.append(stock_payoff)
            
            trading_profit = sum(stock_profit)
  
            profit.append(trading_profit)
    
            open_num.append(trade)
        
            if trade == 0:           # 如果都沒有開倉，報酬率為0
            
                rt.append(0)
            
            else:                    # 計算平均報酬
            
                rt.append(spread_return/trade)

        profit = pd.DataFrame(profit)      
        
        if self.tax_cost == 0:
            
            profit.columns = ["profit without cost"]
            
        else:
            
            profit.columns = ["profit"]
        
        open_num = pd.DataFrame(open_num)   ; open_num.columns = ["open number"]
        rt = pd.DataFrame(rt)               ; rt.columns = ["return"]
    
        back_test = pd.concat([profit,open_num,rt],axis=1)
    
        return back_test
    
    # pair 進行回測--------------------------------------------------------------------------------------------------
    
    def append_backtest_Result(self,result):
        
        self.stock1_name.extend(result[0])
        self.stock2_name.extend(result[1])
        self.profit.extend(result[2])
        self.open_num.extend(result[3])
        self.rt.extend(result[4])
        self.std.extend(result[5])
        self.skew.extend(result[6])
        self.timetrend.extend(result[7])
        self.pos.extend(result[8])
        
        #print(result)
    
    def backtest_table(self):
        
        if self.flag == 1:
            
            from trading_period import pairs
        
        else:
        
            from trading_period_with_timetrend import pairs
        
        #pool = Pool(processes=12) 
        result = []
        n = len(self.table)
        for j in range(n):
            
            y = self.table.iloc[j,:]
            #pool.apply_async( pairs , ( j , self.formate_time , y , self.min_data , self.tick_data , 
            #                           self.open_time , self.close_time ,self.stop_loss_time , self.day1 , self.maxi , 
            #                           self.tax_cost , self.cost_gate , self.capital , self.model_CNN,) , callback=self.append_backtest_Result )
            
            result.append(pairs(j , self.formate_time , y , self.min_data , self.tick_data , self.open_time , self.close_time ,
                                self.stop_loss_time , self.day1 , self.maxi , self.tax_cost , self.cost_gate , self.capital , self.model_CNN))
            
        #pool.close()
        #pool.join()
        '''
        self.stock1_name = pd.DataFrame(self.stock1_name)  ; self.stock1_name.columns = ["stock1"]
        self.stock2_name = pd.DataFrame(self.stock2_name)  ; self.stock2_name.columns = ["stock2"]
        self.profit = pd.DataFrame(self.profit)       
        
        if self.tax_cost == 0:
            
            self.profit.columns = ["profit_without_cost"]
            
        else:
            
            self.profit.columns = ["profit"]
            
        self.open_num = pd.DataFrame(self.open_num)   ; self.open_num.columns = ["open number"]
        self.rt = pd.DataFrame(self.rt)               ; self.rt.columns = ["return"]
        self.std = pd.DataFrame(self.std)             ; self.std.columns = ["std_1"]
        self.skew = pd.DataFrame(self.skew)           ; self.skew.columns = ["skew_1"]
        self.timetrend = pd.DataFrame(self.timetrend) ; self.timetrend.columns = ["time_trend"]
        
        self.pos = pd.DataFrame(self.pos)             ; self.pos.columns = ["pos"]
        
        back_test = pd.concat([self.stock1_name,self.stock2_name,self.profit,self.open_num,self.rt,self.std,self.skew,self.timetrend,self.pos] , axis = 1)
        
        del self.stock1_name
        del self.stock2_name
        del self.profit
        del self.open_num
        del self.rt
        del self.std
        del self.skew
        del self.timetrend
        del self.pos
        '''
        
        
        
        return result #back_test
        