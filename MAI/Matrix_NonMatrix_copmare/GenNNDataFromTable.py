# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:44:53 2020

@author: MAI
"""

import time
import numpy as np
import pandas as pd
import os
from integer import num_weight
from statsmodels.tsa.stattools import adfuller
from InitialUsingMatrix import *

def Where_cross_threshold(trigger_spread, threshold, add_num):
    #initialize array
    check = np.zeros(trigger_spread.shape)
    #put on the condiction
    check[(trigger_spread - threshold) > 0] = add_num
    check[:,0] = check[:,1]
    #Open_trigger_array
    check = check[:,1:] - check[:,:-1]
    return check

def tax(payoff,rate):
    tax_price = payoff * (1 - rate * (payoff > 0))
    return tax_price

def PdCombine(Alltable,table):
    if len(Alltable) == 0:
        Alltable = table.copy()
    else:
        Alltable = pd.concat([Alltable,table])
    return Alltable

def NpCombine(All,new):
    if len(All) == 0:
        All = new
    else:
        All = np.concatenate([All,new])
    return All

#initialize
tick = True                # using tick data or not
EarlyStop = 60              # After how many minutes, we stop opening spread 
TradLen = 0*(1+tick*11)               # Trading lenth
Cost_threshold = False       # whether to use cost threshold
Adf = False                  # Whether to use Adf-check before open
open_thres = 2           # Open threshold (std)
close_thres = 0          # Close threshold (std)
cost_gate = 0.005           # cost gate threshold (percent)
form_del_min = 16           # how many minutes are deleted in formation period
capital = 5000              # capital
maxi = 5                    # restrict the number of ticket we could buy for a stock
OpenSlipTick = 0            # slip how many tick when open
CloseSlipTick = 0           # slip how many tick when close
TpLen = 100 * ( 1 + tick*11 )

tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20191128\dilated_cnn_min_open0.85_close0.75'
test_dataPath = r'C:\Users\MAI\Desktop\2015\averageprice'
test_volumePath = r'C:\Users\MAI\Desktop\min-API\accumulate_volume'
tick_trigger_dataPath = r'D:\tick(secs)\2015'
origin_trigger_dataPath = r'C:\Users\MAI\Desktop\2015\minprice'


years = ["2015"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
#months = ["12"]
days = ["01","02","03","04","05","06","07","08","09","10",
       "11","12","13","14","15","16","17","18","19","20",
       "21","22","23","24","25","26","27","28","29","30",
       "31"]

for year in years:
    for month in months:
        print("Now import: ",month,"-th month")
        for day in days:
            try:
                #Read data
                table = pd.read_csv(os.path.join(tablePath,'{}{}{}_table.csv'.format(year,month,day)))
                test_data = pd.read_csv(os.path.join(test_dataPath,'{}{}{}_averagePrice_min.csv'.format(year,month,day)))
                test_data = test_data.iloc[form_del_min:,:]
                test_data = test_data.reset_index(drop=True)
                test_volume = pd.read_csv(os.path.join(test_volumePath,'{}{}{}_volume.csv'.format(year,month,day)))
                if tick:
                    trigger_data = pd.read_csv(os.path.join(tick_trigger_dataPath,'{}{}{}_tick_stock.csv'.format(year,month,day)))
                    sec_5 = np.arange(9000+60*form_del_min,trigger_data.shape[0]-1,5)
                    trigger_data = trigger_data.iloc[sec_5,:]
                    trigger_data = trigger_data.reset_index(drop=True)
                else:
                    trigger_data = pd.read_csv(os.path.join(origin_trigger_dataPath,'{}{}{}_min_stock.csv'.format(year,month,day)))
                    trigger_data = trigger_data.iloc[(149+form_del_min):,:]
                    trigger_data = trigger_data.reset_index(drop=True)
                
            except:
                continue
            
            if Cost_threshold:
                #Delete the spread that surely unprofitable 
                std = np.array(table.stdev)
                table = table.iloc[(open_thres + close_thres)*std > cost_gate,:]
            
            #Preprocessing data --> VECM spread
            stock1_name = table.stock1.astype('str',copy=False)
            stock2_name = table.stock2.astype('str',copy=False)
            drop = []
            for i in range(len(stock1_name)):
                q1 = stock1_name.iloc[i] not in trigger_data.columns
                q2 = stock2_name.iloc[i] not in trigger_data.columns
                q3 = stock1_name.iloc[i] not in test_volume.columns
                q4 = stock2_name.iloc[i] not in test_volume.columns
                if q1 or q2 or q3 or q4 :
                    drop.append(i)
                    #print('stock1',stock1_name.iloc[i])
                    #print('stock2',stock2_name.iloc[i])
            stock1_name = stock1_name.drop(stock1_name.index[drop])
            stock2_name = stock2_name.drop(stock2_name.index[drop])
            table = table.drop(table.index[drop])
            
            trigger_stock1 = np.array(trigger_data[stock1_name].T)
            trigger_stock2 = np.array(trigger_data[stock2_name].T)
            test_stock1 = np.array(test_data[stock1_name].T)
            test_stock2 = np.array(test_data[stock2_name].T)
            w1 = np.expand_dims(np.array(table.w1),axis=1)
            w2 = np.expand_dims(np.array(table.w2),axis=1)
            trigger_spread = w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2)
            test_spread = w1 * np.log(test_stock1) + w2 * np.log(test_stock2)
            
            #Preprocessing Volume
            test_vol_stock1 = np.array(test_volume[stock1_name].T)
            test_vol_stock2 = np.array(test_volume[stock2_name].T)
            test_v1_max = np.max(test_vol_stock1[:,:150],axis=1)
            test_v2_max = np.max(test_vol_stock2[:,:150],axis=1)
            test_v1_sum = np.sum(test_vol_stock1[:,:150],axis=1)
            test_v2_sum = np.sum(test_vol_stock2[:,:150],axis=1)
            
            if tick:
                trigger_stock1 = trigger_stock1[:,:-48]
                trigger_stock2 = trigger_stock2[:,:-48]
                trigger_spread = trigger_spread[:,:-48]
            else:
                trigger_stock1 = trigger_stock1[:,:-5]
                trigger_stock2 = trigger_stock2[:,:-5]
                trigger_spread = trigger_spread[:,:-5]
            test_spread = test_spread[:,:-5]
            test_stock1 = test_stock1[:,:-5]
            test_stock2 = test_stock2[:,:-5]
                
            #reconstruct std and mu
            std = np.array(table.stdev)
            mu = np.array(table.mu)
            
            ######Matrix vesion######
            ##配合原交易程式碼，我們只從Price[formation+1],Price[formation+2]之後
            ##才開始觀察，而非Price[formation],Price[formation+1]便開始
            
            #threshold
            up_open = np.expand_dims(mu+open_thres*std,axis = 1)
            down_open = np.expand_dims(mu-open_thres*std,axis = 1)
            up_close = np.expand_dims(mu-close_thres*std,axis = 1)
            down_close = np.expand_dims(mu+close_thres*std,axis = 1)
         
            #Where_cross_threshold
            OpCheck_up = Where_cross_threshold(trigger_spread, up_open, 1)
            OpCheck_down = Where_cross_threshold(trigger_spread, down_open, 3)
            ClCheck_up = Where_cross_threshold(trigger_spread, up_close, 1)
            ClCheck_down = Where_cross_threshold(trigger_spread, down_close, 3)
           
            #Combine open trigger array
            OpMix = OpCheck_up+OpCheck_down
            arg_mix = np.argmax(OpMix!=0,axis = 1)
            
            #Open_test_array
            if tick:
                arg_test = arg_mix//12
            else:
                arg_test = arg_mix.copy()
            
            #EarlyStop
            if EarlyStop:
                arg_mix[arg_test>EarlyStop] = 0
                arg_test[arg_test>EarlyStop] = 0
            
            #if not open, return "-1"
            Label = np.zeros(len(arg_mix))
            ClPos = np.ones(len(arg_mix))*(-7)
            ClPos = np.int64(ClPos)
            LongOrShort = np.zeros(len(arg_mix))
            record = np.zeros(len(arg_mix))
            for i in range(len(arg_mix)):
                condiction = abs(OpMix[i,arg_mix[i]])
                if condiction == 1:
                    LongOrShort[i] = -1
                    Pos = np.argmax(ClCheck_up[i,arg_mix[i]:]!=0)
                    if Pos == 0 or (arg_mix[i]+Pos) > (TpLen-3):
                        if TradLen and arg_mix[i] < (TpLen-TradLen-2):
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = -3
                            record[i] = -4
                    else:
                        if TradLen and Pos > TradLen:
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = arg_mix[i] + Pos
                            record[i] = 666
                            Label[i] = 1
                elif condiction == 3:
                    LongOrShort[i] = 1
                    Pos = np.argmax(ClCheck_down[i,arg_mix[i]:]!=0)
                    if Pos == 0 or (arg_mix[i]+Pos) > (TpLen-3):
                        if TradLen and arg_mix[i] < (TpLen-TradLen-2):
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = -3
                            record[i] = -4
                    else:
                        if TradLen and Pos > TradLen:
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = arg_mix[i] + Pos
                            record[i] = 666
                            Label[i] = 1
                elif condiction == 0:
                    Label[i] = -1
                else:
                    arg_mix[i] = 0
                    arg_test[i] = 0
                    #print("Error Condiction: ",condiction)
            #Close_test_array
            if tick:
                ClPos_test = ClPos//12
            else:
                ClPos_test = ClPos.copy()
            
            #trigger price record
            Open_price = np.zeros([len(arg_mix),3])
            Close_price = np.zeros([len(ClPos),3])
            for i in range(len(arg_mix)):
                Open_price[i,0] = trigger_stock1[i,(arg_mix[i]+1+OpenSlipTick)]
                Open_price[i,1] = trigger_stock2[i,(arg_mix[i]+1+OpenSlipTick)]
                Open_price[i,2] = trigger_spread[i,(arg_mix[i]+1+OpenSlipTick)]
                Close_price[i,0] = trigger_stock1[i,(ClPos[i]+1+CloseSlipTick)]
                Close_price[i,1] = trigger_stock2[i,(ClPos[i]+1+CloseSlipTick)]
                Close_price[i,2] = trigger_spread[i,(ClPos[i]+1+CloseSlipTick)] 
            
            table['date'] = ['{}{}{}'.format(year,month,day)] * len(table)
            Alltable = PdCombine(Alltable,table)
            Allarg_test = NpCombine(Allarg_test,arg_test)
            Allarg_mix = NpCombine(Allarg_mix,arg_mix)
            AllClPos = NpCombine(AllClPos,ClPos)
            AllClPos_test = NpCombine(AllClPos_test,ClPos_test)
            AllLongOrShort = NpCombine(AllLongOrShort,LongOrShort)
            AllOpen_price = NpCombine(AllOpen_price,Open_price)
            AllClose_price = NpCombine(AllClose_price,Close_price)
            Alltest_spread = NpCombine(Alltest_spread,test_spread)
            Alltest_stock1 = NpCombine(Alltest_stock1,test_stock1)
            Alltest_stock2 = NpCombine(Alltest_stock2,test_stock2)
            Alltest_vol_stock1 = NpCombine(Alltest_vol_stock1,test_vol_stock1)
            Alltest_vol_stock2 = NpCombine(Alltest_vol_stock2,test_vol_stock2)
            Allrecord = NpCombine(Allrecord,record)
            AllLabel = NpCombine(AllLabel,Label)
            Alltest_v1_max = NpCombine(Alltest_v1_max,test_v1_max)
            Alltest_v2_max = NpCombine(Alltest_v2_max,test_v2_max)
            Alltest_v1_sum = NpCombine(Alltest_v1_sum,test_v1_sum)
            Alltest_v2_sum = NpCombine(Alltest_v2_sum,test_v2_sum)

        #change
        table = Alltable
        arg_test = Allarg_test
        arg_mix = Allarg_mix
        ClPos = AllClPos
        ClPos_test = AllClPos_test
        LongOrShort = AllLongOrShort
        Open_price = AllOpen_price
        Close_price = AllClose_price
        test_spread = Alltest_spread
        test_stock1 = Alltest_stock1
        test_stock2 = Alltest_stock2
        test_vol_stock1 = Alltest_vol_stock1
        test_vol_stock2 = Alltest_vol_stock2
        record = Allrecord
        Label = AllLabel
        test_v1_max = Alltest_v1_max
        test_v2_max = Alltest_v2_max
        test_v1_sum = Alltest_v1_sum
        test_v2_sum = Alltest_v2_sum
        
        ##Open Test
            
        #ADF Test
        Open_table_index = []
        int_w = []
        for i in range(len(arg_mix)):
            if arg_mix[i] != 0:
                w1 , w2 = num_weight(table.w1.iloc[i], table.w2.iloc[i],
                                     Open_price[i,0], Open_price[i,1], maxi, capital)
                if Adf:            
                    if tick:
                        ADF_spread = w1 * np.log( test_stock1[i,:(150+arg_test[i])] ) +\
                                     w2 * np.log( test_stock2[i,:(150+arg_test[i])] )
                    else:   #此處僅為表示tick與非tick算式一樣但想法上有本質上的不同
                        ADF_spread = w1 * np.log( test_stock1[i,:(150+arg_test[i])] ) +\
                                     w2 * np.log( test_stock2[i,:(150+arg_test[i])] )
                    if adfuller( ADF_spread , regression='c' )[1] <= 0.05:
                        Open_table_index.append(i)
                        int_w.append([w1,w2])
                else:
                    Open_table_index.append(i)
                    int_w.append([w1,w2])

        Open_table_index = np.array(Open_table_index)
        #delete useless table row
        table = table.iloc[Open_table_index,:]
        arg_test = arg_test[Open_table_index]
        arg_mix = arg_mix[Open_table_index]
        ClPos = ClPos[Open_table_index]
        ClPos_test = ClPos_test[Open_table_index]
        LongOrShort = LongOrShort[Open_table_index]
        Open_price = Open_price[Open_table_index]
        Close_price = Close_price[Open_table_index]
        test_spread = test_spread[Open_table_index,:]
        test_stock1 = test_stock1[Open_table_index,:]
        test_stock2 = test_stock2[Open_table_index,:]
        test_vol_stock1 = test_vol_stock1[Open_table_index,:]
        test_vol_stock2 = test_vol_stock2[Open_table_index,:]
        record = record[Open_table_index]
        Label = Label[Open_table_index]
        int_w = np.array(int_w)
        
        table = table.reset_index(drop = True)
        
        prediction = np.ones(len(arg_test))
        pair_pos = np.arange(len(arg_test))
        
        #Final close position
        EndPrice = np.zeros([len(arg_test),3])
        for i in range(len(pair_pos)):
            if i == 0:
                condiction = sum(prediction[0:pair_pos[i]] == 0)
                last = 0
            else:
                condiction = sum(prediction[pair_pos[i-1]:pair_pos[i]] == 0)
                last = pair_pos[i-1]
            if condiction == 0:
                EndPrice[i,:] = Close_price[i,:]
            elif condiction >0:
                print('why you detect?')
            else:
                print("Close condiction error.")
        ###Trading###
        
        #Reward
        Reward_0 = np.zeros(len(arg_test))
        Reward_015 = np.zeros(len(arg_test))
        Reward_03 = np.zeros(len(arg_test))
        
        #Trading_cost
        for i in range(len(arg_test)):
            OpenS1Payoff = -LongOrShort[i] * Open_price[i,0] * int_w[i,0]
            OpenS2Payoff = -LongOrShort[i] * Open_price[i,1] * int_w[i,1]
            CloseS1Payoff = LongOrShort[i] * EndPrice[i,0] * int_w[i,0]
            CloseS2Payoff = LongOrShort[i] * EndPrice[i,1] * int_w[i,1]
            Reward_0[i] = OpenS1Payoff + OpenS2Payoff + CloseS1Payoff + CloseS2Payoff
            Reward_015[i] = tax(OpenS1Payoff,0.0015) + tax(OpenS2Payoff,0.0015) +\
                        tax(CloseS1Payoff,0.0015) + tax(CloseS2Payoff,0.0015)
            Reward_03[i] = tax(OpenS1Payoff,0.003) + tax(OpenS2Payoff,0.003) +\
                        tax(CloseS1Payoff,0.003) + tax(CloseS2Payoff,0.003)
        output = np.zeros([2*len(arg_test),610])
        for i in range(len(arg_test)):
            output[2*i,10:160] = test_stock1[i,arg_test[i]:(150+arg_test[i])]
            output[(2*i+1),10:160] = test_stock2[i,arg_test[i]:(150+arg_test[i])]
            output[2*i,160:310] = test_vol_stock1[i,arg_test[i]:(150+arg_test[i])]
            output[(2*i+1),160:310] = test_vol_stock2[i,arg_test[i]:(150+arg_test[i])]
            output[2*i,310:460] = test_stock1[i,:150]
            output[(2*i+1),310:460] = test_stock2[i,:150]
            output[2*i,460:610] = test_vol_stock1[i,:150]
            output[(2*i+1),460:610] = test_vol_stock2[i,:150]
            output[2*i,0] = Label[i]
            output[(2*i+1),0] = Label[i]
            output[2*i,1] = arg_test[i]
            output[(2*i+1),1] = arg_test[i]
            output[2*i,2] = ClPos_test[i]
            output[(2*i+1),2] = ClPos_test[i]
            output[2*i,3] = np.array(table.w1.iloc[i])
            output[(2*i+1),3] = np.array(table.w2.iloc[i])
            output[2*i,4] = int_w[i,0]
            output[(2*i+1),4] = int_w[i,1]
            output[2*i,5] = np.array(table.mu.iloc[i])
            output[(2*i+1),5] = np.array(table.mu.iloc[i])
            output[2*i,6] = np.array(table.stdev.iloc[i])
            output[(2*i+1),6] = np.array(table.stdev.iloc[i])
            output[2*i,7] = Reward_03[i]
            output[(2*i+1),7] = Reward_03[i]
            output[2*i,8] = test_v1_max[i]
            output[(2*i+1),8] = test_v2_max[i]
            output[2*i,9] = test_v1_sum[i]
            output[(2*i+1),9] = test_v2_sum[i]
        save_path = 'D:/mindata_new({},{})_E70L30Cut16NoAdf'.format(str(open_thres),str(close_thres))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path+'/{}{}data.npy'.format(year,month),output)
        Alltable = []
        Allarg_test = []
        Allarg_mix = []
        AllClPos = []
        AllClPos_test = []
        AllCNNDetPos = []
        AllCNNDetPos_test = []
        AllLongOrShort = []
        AllOpen_price = []
        AllClose_price = []
        AllCNNBreak_price = []
        Alltest_spread = []
        Alltest_stock1 = []
        Alltest_stock2 = []
        Alltest_vol_stock1 = []
        Alltest_vol_stock2 = []
        Allrecord = []
        AllLabel = []
        Alltest_v1_max = []
        Alltest_v2_max = []
        Alltest_v1_sum = []
        Alltest_v2_sum = []
        
