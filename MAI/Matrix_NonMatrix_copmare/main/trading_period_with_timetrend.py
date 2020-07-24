# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:08:05 2018

@author: chuchu0936
"""

#import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from cost import tax , slip 
from integer import num_weight
from scipy.stats import skew
from MTSA import fore_chow
import pandas as pd
import numpy as np

# 標準差倍數當作停損門檻(滑價＋交易稅)-------------------------------------------------------------------------------

def pairs( pair , formate_time , table , min_data , tick_data , open_time , close_time , stop_loss_time , day1 , maxi , tax_cost , cost_gate , capital , model):
    
    table = pd.DataFrame(table).T
    
    # 波動太小的配對不開倉
    if (open_time + close_time) * table.stdev[pair] < cost_gate or table.model_type[pair] != 'model5': 
        
        trading_profit = 0
        
        trade = 0
        
        local_profit = 0
        local_open_num = 0 
        local_rt = 0 
        local_std = 0 
        local_skew = 0 
        local_timetrend = 0 
        position = 0
        
        return  [ local_profit , local_open_num , local_rt , local_std , local_skew , local_timetrend , position ]
    
    min_price = day1
    #min_price = min_price.dropna(axis = 1)
    #min_price.index  = np.arange(0,len(min_price),1)
    
    #num = np.arange(0,len(table),1)
    
    t = formate_time                                           # formate time
    
    local_open_num = []
    local_profit = []
    local_rt = []
    #for pair in num:
    
    #spread = table.w1[pair] * np.log(min_data[ table.stock1[pair] ]) + table.w2[pair] * np.log(min_data[ table.stock2[pair] ])
    spread = table.w1[pair] * np.log(tick_data[ table.stock1[pair] ]) + table.w2[pair] * np.log(tick_data[ table.stock2[pair] ])
    
    if table.model_type[pair] == 'model4':
        
        x = np.arange(0,t)
        
        # 算出spread在formation period的斜率
        y = table.w1[pair] * np.log(min_price[table.stock1[pair]].iloc[0:t]) + table.w2[pair] * np.log(min_price[table.stock2[pair]].iloc[0:t])
        b1 , b0 = np.polyfit(x,y,1)
        
        # 算trading period 的序列
        x = np.arange(t,len(min_price))
        
        trend_line = x*b1 + b0
        spread = spread - trend_line
        
    else:
        
        b1 = 0
    
    up_open = table.mu[pair] + table.stdev[pair] * open_time                      # 上開倉門檻
    down_open = table.mu[pair] - table.stdev[pair] * open_time                    # 下開倉門檻
        
    stop_loss = table.stdev[pair] * stop_loss_time                                # 停損門檻
        
    close = table.mu[pair]                                                        # 平倉(均值)
        
    M = round( 1 / table.zcr[pair] )                                              # 平均持有時間
        
    trade = 0                                                                     # 計算開倉次數
    
    break_point = 0
    
    #discount = 1
    
    position = 0                                                                  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    
    #model=load_model('model.h5')
    #model.summary()
    
    pos = []
    stock1_profit = []
    stock2_profit = []
    for i in range( len(spread)-2 ):
        
        stock1_seq = min_price[ table.stock1[pair] ].loc[0:t+i]
        stock2_seq = min_price[ table.stock2[pair] ].loc[0:t+i]
        
        if position == 0 and len(spread)-i > M :                                   # 之前無開倉
        
            if ( spread[i] - up_open ) * ( spread[i+1] - up_open ) < 0 :
                    
                # 資金權重轉股票張數，並整數化
                w1 , w2 = num_weight( table.w1[pair] , table.w2[pair] , tick_data[table.stock1[pair]][(i+1)] , tick_data[table.stock2[pair]][(i+1)] , maxi , capital )          
                    
                spread1 = w1 * np.log( stock1_seq ) + w2 * np.log( stock2_seq )
                    
                if adfuller( spread1 , regression='ct' )[1] > 1 or b1 > 0:                         # spread平穩才開倉 & 趨勢向上不開空倉
                        
                    position = 0
                        
                    stock1_payoff = 0
                        
                    stock2_payoff = 0
                        
                else:
                    
                    position = -1
                    
                    stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+1)] , table.w1[pair] )
                    
                    stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+1)] , table.w2[pair] )
                    
                    stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )  # 計算交易成本
                    
                    down_open = table.mu[pair] - table.stdev[pair] * close_time
                    
                    trade = trade + 1
                
            elif ( spread[i] - down_open ) * ( spread[i+1] - down_open ) < 0 :
                    
                # 資金權重轉股票張數，並整數化
                w1 , w2 = num_weight( table.w1[pair] , table.w2[pair] , tick_data[table.stock1[pair]][(i+1)] , tick_data[table.stock2[pair]][(i+1)] , maxi , capital )          
                    
                spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                    
                if adfuller( spread1 , regression='ct' )[1] > 1 or b1 < 0:                          # spread平穩才開倉 & 趨勢向下不開多倉
                        
                    position = 0
                        
                    stock1_payoff = 0
                        
                    stock2_payoff = 0
                        
                else:
                    
                    position = 1
                    
                    stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+1)] , -table.w1[pair] )
                    
                    stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+1)] , -table.w2[pair] )
                        
                    stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )   # 計算交易成本
                    
                    up_open = table.mu[pair] + table.stdev[pair] * close_time
                    
                    trade = trade + 1
                    
            else: 
                    
                position = 0
        
                stock1_payoff = 0
            
                stock2_payoff = 0
        
        elif position == -1:                                                                         # 之前有開空倉，平空倉
            
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            
            num = fore_chow( min_price[table.stock1[pair]].loc[0:t] , min_price[table.stock2[pair]].loc[0:t] , stock1_seq , stock2_seq , table.model_type[pair] )
            
            if num == 0:
                
                break_point = 0
                
            else: # num == 1
                
                break_point = break_point + num
            
            
            
            if ( spread[i] - down_open ) * ( spread[i+1] - down_open ) < 0 :
        
                position = 666                                                                         # 平倉
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+1)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+1)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )       # 計算交易成本
                
                down_open = table.mu[pair] - table.stdev[pair] * open_time
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif ( spread[i] - (close + stop_loss) ) * ( spread[i+1] - (close + stop_loss) ) < 0 :
                
                position = -2                                                                                   # 碰到停損門檻，強制平倉
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+1)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+1)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )       # 計算交易成本
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif adfuller( spread1 )[1] > 0.05 :
                
                #position = -3                                                                                    # 出現單跟，強制平倉
            
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+1)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+1)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif break_point == 3 :
                
                position = -3                                                                                    # 結構性斷裂，強制平倉
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+1)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+1)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif i == (len(spread)-3) :                                                                          # 回測結束，強制平倉
            
                position = -4
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][len(tick_data)-1] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][len(tick_data)-1] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )         # 計算交易成本
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            else: 
            
                position = -1
        
                stock1_payoff = 0
            
                stock2_payoff = 0
        
        elif position == 1:                                                                                        # 之前有開多倉，平多倉
            
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            
            num = fore_chow( min_price[table.stock1[pair]].loc[0:t] , min_price[table.stock2[pair]].loc[0:t] , stock1_seq , stock2_seq , table.model_type[pair] )
            
            if num == 0:
                
                break_point = 0
                
            else: # num == 1
                
                break_point = break_point + num
            
            
            
            if ( spread[i] - up_open ) * ( spread[i+1] - up_open ) < 0 :
        
                position = 666                                                                                       # 平倉
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+1)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+1)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )           # 計算交易成本
                
                up_open = table.mu[pair] + table.stdev[pair] * open_time
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif ( spread[i] - (close - stop_loss) ) * ( spread[i+1] - (close - stop_loss) ) < 0 :
                
                position = -2                                                                                       # 碰到停損門檻，強制平倉
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+1)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+1)] , table.w2[pair] )
                
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )           # 計算交易成本
                
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif adfuller( spread1 )[1] > 0.05 :
                
                #position = -2                                                                                    # 出現單跟，強制平倉
            
                #stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+1)] , table.w1[pair] )
                
                #stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+1)] , table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif break_point == 3 :
                
                position = -3                                                                                        # 結構性斷裂，強制平倉
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+1)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+1)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )            # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif i == (len(spread)-3) :                                                                              # 回測結束，強制平倉
            
                position = -4
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][len(tick_data)-1] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][len(tick_data)-1] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )            # 計算交易成本
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            else: 
            
                position = 1
        
                stock1_payoff = 0
                    
                stock2_payoff = 0
            
        else:
                
            if position == -2 or position == -3 or position == -4 or position == 666:
                
                stock1_payoff = 0
                
                stock2_payoff = 0
                    
            else:
        
                position = 0                                                                         # 剩下時間少於預期開倉時間，則不開倉，避免損失
        
                stock1_payoff = 0
                
                stock2_payoff = 0
                    
        pos.append(position)
        
        stock1_profit.append(stock1_payoff)
            
        stock2_profit.append(stock2_payoff)
            
    trading_profit = sum(stock1_profit) + sum(stock2_profit)
    
    if trading_profit != 0 and position == 666:
        
        position = 666
        
    #local_profit.append(trading_profit)
    local_profit = trading_profit
        
    #local_open_num.append(trade)
    local_open_num = trade
        
    if trade == 0:            # 如果都沒有開倉，則報酬為0
        
        #local_rt.append(0)
        #local_std.append(0)
        #local_skew.append(0)
        #local_timetrend.append(0)
        
        local_rt = 0
        local_std = 0
        local_skew = 0
        local_timetrend = 0
        position = 0
        
    else:                     # 計算平均報酬
        
        #local_rt.append(trading_profit/(capital*trade) )
        
        spread2 = w1 * np.log(min_price[table.stock1[pair]].iloc[0:t]) + w2 * np.log(min_price[table.stock2[pair]].iloc[0:t])
        
        #local_std.append(np.std(spread2))
        #local_skew.append(skew(spread2))
        
        x = np.arange(0,t)
        b1 , b0 = np.polyfit(x,spread2,1)
        
        #local_timetrend.append(b1)
        
        local_rt = trading_profit/(capital*trade)
        local_std = np.std(spread2)
        local_skew = skew(spread2)
        local_timetrend = b1
        
        #x = np.arange(0,121)
        #plt.plot(spread)
        #plt.axhline(y=close,color='r')
        #plt.axhline(y=up_open,color='r')
        #plt.axhline(y=down_open,color='r')
        #plt.axhline(y=close+stop_loss,color='green')
        #plt.axhline(y=close-stop_loss,color='green')
            
        #bp = np.array(np.where( np.array(pos) == -3 ))
            
        #if bp.size != 0:
                
            #plt.axvline(x=bp[0][0],color='green')
            
        #plt.show()
        
        #print([table.stock1[pair],table.stock2[pair],trading_profit])
        
    #posi = pos[len(spread)-2]
    
    '''
    if tax_cost == 0:
    
        local_profit = pd.DataFrame(local_profit)       ; local_profit.columns = ["profit without cost"]
        
    else:
        
        local_profit = pd.DataFrame(local_profit)       ; local_profit.columns = ["profit"]
        
    local_open_num = pd.DataFrame(local_open_num)   ; local_open_num.columns = ["open number"]
    local_rt = pd.DataFrame(local_rt)               ; local_rt.columns = ["return"]
    
    #back_test = pd.concat([local_profit,local_open_num,local_rt],axis=1)
    '''
    
    return [ local_profit , local_open_num , local_rt , local_std , local_skew , local_timetrend , position ]#table.stock1 , table.stock2 , local_profit , local_open_num , local_rt , local_std , local_skew , local_timetrend #, 0
    
    
    


