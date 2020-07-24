# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 18:08:05 2018

@author: chuchu0936
"""

#from formation_period import formation_period

from statsmodels.tsa.stattools import adfuller
# from Predict_Client import send_request
from .cost import tax , slip 
from .integer import num_weight
from .MTSA import fore_chow , spread_chow
#import tensorflow
#from keras.models import load_model
import pandas as pd
import numpy as np

# 標準差倍數當作停損門檻(滑價＋交易稅)-------------------------------------------------------------------------------

def pairs( pair , formate_time , table , min_data , tick_data , open_time , stop_loss_time , day1 , maxi , tax_cost , capital ):
    
    table = pd.DataFrame(table).T
    
    min_price = day1

    #min_price = min_price.dropna(axis = 1)
    #min_price.index  = np.arange(0,len(min_price),1)
    
    #num = np.arange(0,len(table),1)
    
    t = formate_time                                           # formate time
    
    local_open_num = []
    local_profit = []
    local_rt = []

    trade_history = []
    #for pair in num:
    
    spread = table.w1[pair] * np.log(min_data[ table.stock1[pair] ]) + table.w2[pair] * np.log(min_data[ table.stock2[pair] ])
    
    up_open = table.mu[pair] + table.stdev[pair] * open_time                      # 上開倉門檻
    down_open = table.mu[pair] - table.stdev[pair] * open_time                    # 下開倉門檻
        
    stop_loss = table.stdev[pair] * stop_loss_time                                # 停損門檻
        
    close = table.mu[pair]                                                        # 平倉(均值)
        
    M = round( 1/table.zcr[pair] )  if (len(spread) >= 115 and table.zcr[pair] != 0 ) else 0                                  # 平均持有時間
        
    trade = 0                                                                     # 計算開倉次數
    #discount = 1
    
    position = 0                                                                  # 持倉狀態，1:多倉，0:無倉，-1:空倉，-2：強制平倉
    
    #model=load_model('model.h5')
    #model.summary()
    
    pos = []
    stock1_profit = []
    stock2_profit = []
    no_more = False
    trade_status = ""
    for i in range( len(spread)-2 ):
        trade_status = str(min_price["mtimestamp"][t+i]) + "/"
        stock1_seq = min_price[ table.stock1[pair] ].loc[0:t+i]
        stock2_seq = min_price[ table.stock2[pair] ].loc[0:t+i]
        
        if position == 0 and len(spread)-i > M :                                   # 之前無開倉
            # trade_status += "之前無開倉，且剩餘時間大於平均持有時間，"
            if ( spread[i] - up_open ) * ( spread[i+1] - up_open ) < 0 :
                trade_status += "碰到上開倉門檻, "
                # 資金權重轉股票張數，並整數化
                print(str(table.stock1.values[0]))
                w1 , w2 = num_weight( table.w1[pair] , table.w2[pair] , tick_data[table.stock1[pair]][(i+2)] , tick_data[table.stock2[pair]][(i+2)] , maxi , capital )          
                do_w1 = "買進" if w1 > 0 else "放空"
                do_w2 = "買進" if w2 > 0 else "放空"
                spread1 = w1 * np.log( stock1_seq ) + w2 * np.log( stock2_seq )
                
                # if adfuller( spread1 , regression='ct' )[1] > 0.05:                                   # spread平穩才開倉
                #     trade_status += "spread不平穩，不開倉"
                #     position = 0
                        
                #     stock1_payoff = 0
                        
                #     stock2_payoff = 0
                        
                # else:
                    
                position = -1
                
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )  # 計算交易成本
                
                trade = trade + 1
                trade_status += ",上開倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)
            
            elif ( spread[i] - down_open ) * ( spread[i+1] - down_open ) < 0 :
                trade_status += "碰到下開倉門檻, "
                # 資金權重轉股票張數，並整數化
                w1 , w2 = num_weight( table.w1[pair] , table.w2[pair] , tick_data[table.stock1[pair]][(i+2)] , tick_data[table.stock2[pair]][(i+2)] , maxi , capital )          
                    
                spread1 = w1 * np.log(stock1_seq) + w2 * np.log(stock2_seq)
                    
                # if adfuller( spread1 , regression='ct' )[1] > 0.05 :                                    # spread平穩才開倉
                #     trade_status += "spread不平穩，不開倉"
                #     position = 0
                        
                #     stock1_payoff = 0
                        
                #     stock2_payoff = 0
                        
                # else:
                    
                position = 1
                
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )   # 計算交易成本
                
                trade = trade + 1
                trade_status += "下開倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)
                    
            else: 
                trade_status += "沒碰到開倉門檻，維持不開倉"
                position = 0
        
                stock1_payoff = 0
            
                stock2_payoff = 0
                continue
        
        elif position == -1:                                                                         # 之前有開空倉，平空倉
            trade_status += "之前有開空倉, "
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            
            #temp=spread1[i+1:t+i+1].reshape(1,150,1)
            #pre=model.predict_classes(temp)
            
            if ( spread[i] - close ) * ( spread[i+1] - close ) < 0 :
                
                position = 0                                                                         # 平倉
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )       # 計算交易成本
                trade_status += "碰到均值，平空倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)
                    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif ( spread[i] - (close + stop_loss) ) * ( spread[i+1] - (close + stop_loss) ) < 0 :
                
                
                position = -2                                                                                   # 碰到停損門檻，強制平倉
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )       # 計算交易成本
                trade_status += "碰到停損門檻，強制平倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif adfuller( spread1 , regression='ct' )[1] > 0.05 :
                
                #position = -3                                                                                    # 出現單跟，強制平倉
            
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif fore_chow( min_price[table.stock1[pair]].loc[0:t] , min_price[table.stock2[pair]].loc[0:t] , stock1_seq , stock2_seq ) == 1 :
                
                position = -3                                                                                    # 結構性斷裂，強制平倉
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                trade_status += "結構性斷裂，強制平倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif spread_chow( spread1 , i ) == 1 :
                
                #position = -2                                                                                    # 結構性斷裂，強制平倉
            
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif send_request( spread1 , 150 , 0.9 ) == 1:
                
                #position = -2                                                                                    # LSTM偵測，強制平倉
                
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif ( (pre!=0) | (pre!=4) ):
                
                #position = -2                                                                                    # CNN偵測，強制平倉
                
                #stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][(i+2)] , -table.w1[pair] )
                
                #stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][(i+2)] , -table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif i == (len(spread)-3) :                                                                          # 回測結束，強制平倉
                
                position = -2
            
                stock1_payoff = -w1 * slip( tick_data[table.stock1[pair]][len(tick_data)-1] , -table.w1[pair] )
                
                stock2_payoff = -w2 * slip( tick_data[table.stock2[pair]][len(tick_data)-1] , -table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )         # 計算交易成本
                trade_status += "回測結束，強制平倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            else: 
                trade_status += "沒碰到均值，維持"
                position = -1
        
                stock1_payoff = 0
            
                stock2_payoff = 0
                continue
        
        elif position == 1:                                                                                        # 之前有開多倉，平多倉
            trade_status += "之前有開多倉, "
            spread1 = table.w1[pair] * np.log(stock1_seq) + table.w2[pair] * np.log(stock2_seq)
            
            #temp=spread1[i+1:t+i+1].reshape(1,150,1)
            #pre=model.predict_classes(temp)
            
            if ( spread[i] - close ) * ( spread[i+1] - close ) < 0 :
                
                position = 0                                                                                       # 平倉
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )           # 計算交易成本
                trade_status += "碰到均值，平多倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif ( spread[i] - (close - stop_loss) ) * ( spread[i+1] - (close - stop_loss) ) < 0 :
                
                
                position = -2                                                                                       # 碰到停損門檻，強制平倉
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )           # 計算交易成本
                trade_status += "碰到停損門檻，強制平倉 {} &nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp{}".format(int(w1), stock1_payoff, int(w2), stock2_payoff)    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif adfuller( spread1 , regression='ct' )[1] > 0.05 :
                
                #position = -3                                                                                    # 出現單跟，強制平倉
            
                #stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                #stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif fore_chow( min_price[table.stock1[pair]].loc[0:t] , min_price[table.stock2[pair]].loc[0:t] , stock1_seq , stock2_seq ) == 1 :
                
                position = -3                                                                                        # 結構性斷裂，強制平倉
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )            # 計算交易成本
                trade_status += "結構性斷裂，強制平倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif spread_chow( spread1 , i ) == 1 :
                
                #position = -2                                                                                        # 結構性斷裂，強制平倉
            
                #stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                #stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )            # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif send_request( spread1 , 150 , 0.9 ) == 1:
                
                #position = -2                                                                                    # LSTM偵測，強制平倉
                
                #stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                #stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            #elif ( (pre!=0) | (pre!=4) ):
                
                #position = -2                                                                                    # CNN偵測，強制平倉
                
                #stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][(i+2)] , table.w1[pair] )
                
                #stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][(i+2)] , table.w2[pair] )
                    
                #stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )        # 計算交易成本
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            elif i == (len(spread)-3) :                                                                              # 回測結束，強制平倉
                
                position = -2
            
                stock1_payoff = w1 * slip( tick_data[table.stock1[pair]][len(tick_data)-1] , table.w1[pair] )
                
                stock2_payoff = w2 * slip( tick_data[table.stock2[pair]][len(tick_data)-1] , table.w2[pair] )
                    
                stock1_payoff , stock2_payoff = tax(stock1_payoff , stock2_payoff , position , tax_cost )            # 計算交易成本
                trade_status += "回測結束，強制平倉 <br>{}&nbsp張&nbsp{}&nbsp {}  &nbsp&nbsp&nbsp  {}&nbsp張&nbsp{}&nbsp  {}".format(int(w1), table.stock1.values[0], stock1_payoff, int(w2), table.stock2.values[0], stock2_payoff)    
                #每次交易報酬做累加(最後除以交易次數做平均)
                
            else: 
                trade_status += "沒碰到均值，維持"
                position = 1
        
                stock1_payoff = 0
                    
                stock2_payoff = 0
                continue
            
        else:
            no_more = True
            if position == -2 or position == -3:
                trade_status += "配對已經不適合交易"
                stock1_payoff = 0
                
                stock2_payoff = 0
                    
            else:
                trade_status += "剩下時間少於預期開倉時間，不開倉，避免損失"
                position = 0                                                                         # 剩下時間少於預期開倉時間，則不開倉，避免損失
        
                stock1_payoff = 0
                
                stock2_payoff = 0
                    
        pos.append(position)
            
        stock1_profit.append(stock1_payoff)
            
        stock2_profit.append(stock2_payoff)
        if(no_more):
            break
        trade_history.append(trade_status)
        # print(trade_status)
    #print(position)
    
    #x = np.arange(0,121)
    #plt.plot(spread)
    #plt.axhline(y=close,color='r')
    #plt.axhline(y=up_open,color='r')
    #plt.axhline(y=down_open,color='r')
    #plt.axhline(y=close+stop_loss,color='green')
    #plt.axhline(y=close-stop_loss,color='green')
            
    #bp = np.array(np.where( pos == -3 ))
            
    #if bp.size != 0:
                
        #plt.axvline(x=bp[0][0],color='green')
            
    #plt.show()
    
    trading_profit = sum(stock1_profit) + sum(stock2_profit)
        
    if 1.2 * table.stdev[pair] < tax_cost:
        
        trading_profit = 0
            
        trade = 0

        trade_history = []
        
    local_profit = trading_profit
    #local_profit = trading_profit
    
    local_open_num = trade
    #local_open_num = trade
        
    if trade == 0:            # 如果都沒有開倉，則報酬為0
        
        local_rt = 0
        #local_rt = 0
        
    else:                     # 計算平均報酬
        
        local_rt = trading_profit/(capital*trade)
        #local_rt = trading_profit/(capital*trade)
        
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
    return  {
        'trade_history' : trade_history , 
        "local_profit" : local_profit , 
        "local_open_num" : local_open_num, 
        "local_rt" : local_rt
    } #, 0
    
    