# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:53:18 2020

@author: MAI
"""

import time
import numpy as np
import pandas as pd
import os
from integer import num_weight
from statsmodels.tsa.stattools import adfuller
#from keras.models import load_model
from InitialUsingMatrix import *
from Matrix_function import Where_cross_threshold,tax,CNN_test,fore_chow
from Matrix_function import Where_threshold
#from vecm import para_vecm

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
tick = True             # using tick data or not
half = False                # using half min test data (please Open the "Tick")
EarlyStop = 40*(1+half)              # After how many minutes, we stop opening spread 
TradLen = 0*(1+tick*11)               # Trading lenth
Cost_threshold = True       # whether to use cost threshold
Adf = False                 # Whether to use Adf-check before open
zcr = False
OnlyIn = False
Emu = True
Estd = True
Ethresh = False
open_thres = 1.5          # Open threshold (std)
close_thres = 0          # Close threshold (std)
forecast = False
forelag5 = False
stoploss_thres = False
cross = True
cost_gate = 0.005           # cost gate threshold (percent)
form_del_min = 16           # how many minutes are deleted in formation period
capital = 50000000              # capital
maxi = 5                    # restrict the number of ticket we could buy for a stock
OpenSlipTick = 0            # slip how many tick when open
CloseSlipTick = 0           # slip how many tick when close
TpLen = 100 * ( 1 + tick*11 )
check_spread = True
save_spread = False
save_path = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\result\O{}C{}E{}.npy'.format(open_thres,close_thres,EarlyStop)

#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\EOrigin_table'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\EOrigin(1)_table'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200602Eorigin'
tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200608Estorigin'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200611ENoNormality'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200619Etestcorrect'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\forelag5_min_open0.85_close0.75'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20191128\cnn_min_open0.85_close0.75_new'

if half:
    test_csv_name = '_half_min'
else:
    test_csv_name = '_averagePrice_min'


years = ['2015']
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
#months = ["03"]
days = ["01","02","03","04","05","06","07","08","09","10",
       "11","12","13","14","15","16","17","18","19","20",
       "21","22","23","24","25","26","27","28","29","30",
       "31"]
#days = ["15"]
#Clock

error_count = 0
start_time = time.time()
for year in years:
    #tablePath = r'C:\Users\MAI\Desktop\trading_result\fore_lag5_half_thres0.45_open0.85close_0.85'
    test_dataPath = r'C:\Users\MAI\Desktop\{}\averageprice'.format(year)
    #test_dataPath = r'C:\Users\MAI\Desktop\half-min\2016\0050'
    test_volumePath = r'C:\Users\MAI\Desktop\min-API\accumulate_volume'
    tick_trigger_dataPath = r'D:\tick(secs)\{}'.format(year)
    origin_trigger_dataPath = r'C:\Users\MAI\Desktop\{}\minprice'.format(year)
    for month in months:
        print("Now import: ",month,"-th month")
        for day in days:
            try:
                #Read data
                
#                if year == '2018' and Emu!=True:
#                    table = pd.read_csv(os.path.join(tablePath,'{}{}{}.csv'.format(year,month,day)))
#                else:
                table = pd.read_csv(os.path.join(tablePath,'{}{}{}_table.csv'.format(year,month,day)))
                test_data = pd.read_csv(os.path.join(test_dataPath,'{}{}{}{}.csv'.format(year,month,day,test_csv_name)))
                test_data = test_data.iloc[form_del_min:,:]
                test_data = test_data.reset_index(drop=True)
                if tick:
                    trigger_data = pd.read_csv(os.path.join(tick_trigger_dataPath,'{}{}{}_tick_stock.csv'.format(year,month,day)))
                    sec_5 = np.arange(9000+60*form_del_min,trigger_data.shape[0]-1,5)
                    trigger_data = trigger_data.iloc[sec_5,:]
                    trigger_data = trigger_data.reset_index(drop=True)
                else:
#                    trigger_data = pd.read_csv(os.path.join(test_dataPath,'{}{}{}{}.csv'.format(year,month,day,test_csv_name)))
                    trigger_data = pd.read_csv(os.path.join(origin_trigger_dataPath,'{}{}{}_min_stock.csv'.format(year,month,day)))
                    trigger_data = trigger_data.iloc[(149+form_del_min):,:]
                    trigger_data = trigger_data.reset_index(drop=True)
                
            except:
                
                continue
            
            if Cost_threshold:
                #Delete the spread that surely unprofitable 
                if Estd:
                    std = np.array(table.Estd)
                else:
                    std = np.array(table.stdev)
                table = table.iloc[(open_thres + close_thres)*std > cost_gate,:]
            if zcr:
                table = table.iloc[np.array(table.zcr<=0.07),:]
                
            if len(table) == 0:
                continue
            
            #Preprocessing data --> VECM spread
            stock1_name = table.stock1.astype('str',copy=False)
            stock2_name = table.stock2.astype('str',copy=False)
            drop = []
            for i in range(len(stock1_name)):
                q1 = stock1_name.iloc[i] not in trigger_data.columns
                q2 = stock2_name.iloc[i] not in trigger_data.columns
                if q1 or q2 :
                    drop.append(i)
                    #print('Date:',month,day)
                    #print('stock1',stock1_name.iloc[i])
                    #print('stock2',stock2_name.iloc[i])
            stock1_name = stock1_name.drop(stock1_name.index[drop])
            stock2_name = stock2_name.drop(stock2_name.index[drop])
            table = table.drop(table.index[drop])
            
            stock1_name = table.stock1.astype('str',copy=False)
            stock2_name = table.stock2.astype('str',copy=False)
            trigger_stock1 = np.array(trigger_data[stock1_name].T)
            trigger_stock2 = np.array(trigger_data[stock2_name].T)
            test_stock1 = np.array(test_data[stock1_name].T)
            test_stock2 = np.array(test_data[stock2_name].T)
            w1 = np.expand_dims(np.array(table.w1),axis=1)
            w2 = np.expand_dims(np.array(table.w2),axis=1)
            trigger_spread = w1 * np.log(trigger_stock1) + w2 * np.log(trigger_stock2)
            test_spread = w1 * np.log(test_stock1) + w2 * np.log(test_stock2)
            
            #Preprocessing Volume
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
            if Estd:
                std = np.array(table.Estd)
            else:
                std = np.array(table.stdev)
            if Emu:
                mu = np.array(table.Emu)
            else:
                mu = np.array(table.mu)
            
            ######Matrix vesion######
            ##配合原交易程式碼，我們只從Price[formation+1],Price[formation+2]之後
            ##才開始觀察，而非Price[formation],Price[formation+1]便開始
            
            
            #threshold
            if Ethresh:
                up_open = np.expand_dims(mu+Ethresh,axis = 1)
                down_open = np.expand_dims(mu-Ethresh,axis = 1)
                up_close = np.expand_dims(mu-close_thres,axis = 1)
                down_close = np.expand_dims(mu+close_thres,axis = 1)
            else:
                up_open = np.expand_dims(mu+open_thres*std,axis = 1)
                down_open = np.expand_dims(mu-open_thres*std,axis = 1)
                up_close = np.expand_dims(mu-close_thres*std,axis = 1)
                down_close = np.expand_dims(mu+close_thres*std,axis = 1)
            Stl_up = np.expand_dims(mu+stoploss_thres*std,axis = 1)
            Stl_down = np.expand_dims(mu-stoploss_thres*std,axis = 1)
            
            #Where_cross_threshold
            if cross:
                OpCheck_up = Where_cross_threshold(trigger_spread, up_open, 1)
                OpCheck_down = Where_cross_threshold(trigger_spread, down_open, 3)
                ClCheck_up = Where_cross_threshold(trigger_spread, up_close, 1)
                ClCheck_down = Where_cross_threshold(trigger_spread, down_close, 3)
                StlCheck_up = Where_cross_threshold(trigger_spread, Stl_up, 1)
                StlCheck_down = Where_cross_threshold(trigger_spread, Stl_down, 3)
                #Check Double Cross and Delete it
                DoubleCrossUp = np.multiply(OpCheck_up,ClCheck_up) 
                DoubleCrossDown = np.multiply(OpCheck_down,ClCheck_down)
                OpCheck_up[DoubleCrossUp!=0] = 0
                OpCheck_down[DoubleCrossDown!=0] = 0
    
                if stoploss_thres:
                    DoubleCrossUp = np.multiply(OpCheck_up,StlCheck_up) 
                    DoubleCrossDown = np.multiply(OpCheck_down,StlCheck_down)
                    OpCheck_up[DoubleCrossUp!=0] = 0
                    OpCheck_down[DoubleCrossDown!=0] = 0
            else:
                OpCheck_up = Where_threshold(trigger_spread, up_open, 1, True)
                OpCheck_down = Where_threshold(trigger_spread, down_open, 3, False)
                ClCheck_up = Where_threshold(trigger_spread, up_close, 1, False)
                ClCheck_down = Where_threshold(trigger_spread, down_close, 3, True)
                StlCheck_up = Where_threshold(trigger_spread, Stl_up, 1, True)
                StlCheck_down = Where_threshold(trigger_spread, Stl_down, 3, False)
    
                if stoploss_thres:
                    DoubleCrossUp = np.multiply(OpCheck_up,StlCheck_up) 
                    DoubleCrossDown = np.multiply(OpCheck_down,StlCheck_down)
                    OpCheck_up[DoubleCrossUp!=0] = 0
                    OpCheck_down[DoubleCrossDown!=0] = 0
            
            
            #Combine open trigger array
            OpMix = OpCheck_up+OpCheck_down
            arg_up = OpCheck_up == -1
            arg_down = OpCheck_down == 3
            arg_open = arg_up + arg_down
            if OnlyIn:
                arg_mix = np.argmax(arg_open!=0,axis = 1)
            else:
                arg_mix = np.argmax(OpMix!=0,axis = 1)
            #Open_test_array
            if tick:
                arg_test = (arg_mix+cross)//(12-6*half)
            else:
                arg_test = arg_mix.copy()
            
            #EarlyStop
            if EarlyStop:
                arg_mix[arg_test>EarlyStop] = 0
                arg_test[arg_test>EarlyStop] = 0
            
            
            #if not open, return "-1"
            ClPos = np.ones(len(arg_mix))*(-7)
            ClPos = np.int64(ClPos)
            LongOrShort = np.zeros(len(arg_mix))
            record = np.zeros(len(arg_mix))
            for i in range(len(arg_mix)):
                condiction = abs(OpMix[i,arg_mix[i]])
                whetherStl = 0
                if condiction == 1:
                    LongOrShort[i] = -1
                    Pos = np.argmax(ClCheck_up[i,arg_mix[i]:]!=0)
                    if stoploss_thres:
                        S = np.argmax(StlCheck_up[i,arg_mix[i]:]!=0)
                        if S!=0:
                            if S<Pos or Pos==0:
                                Pos = S
                                whetherStl = 1
                    if Pos == 0 or (arg_mix[i]+Pos) > (TpLen-3):
                        if Pos == 0 and ClCheck_up[i,arg_mix[i]]!=0:
                            arg_mix[i] = 0
                            arg_test[i] = 0
                            error_count += 1
                            print("Error upOpen Close: ",i)
                            continue
                        if TradLen and arg_mix[i] < (TpLen-TradLen-2):
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = -3
                            record[i] = -4
                            if whetherStl ==1:
                                record[i] = -2
                    else:
                        if TradLen and Pos > TradLen:
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = arg_mix[i] + Pos
                            record[i] = 666
                            if whetherStl ==1:
                                record[i] = -2
                elif condiction == 3:
                    LongOrShort[i] = 1
                    Pos = np.argmax(ClCheck_down[i,arg_mix[i]:]!=0)
                    if stoploss_thres:
                        S = np.argmax(StlCheck_down[i,arg_mix[i]:]!=0)
                        if S!=0:
                            if S<Pos or Pos==0:
                                Pos = S
                                whetherStl = 1
                    if Pos == 0 or (arg_mix[i]+Pos) > (TpLen-3):
                        if Pos==0 and ClCheck_down[i,arg_mix[i]]!=0:
                            arg_mix[i] = 0
                            arg_test[i] = 0
                            error_count += 1
                            print("Error DownOpen Close: ",i)
                            continue
                        if TradLen and arg_mix[i] < (TpLen-TradLen-2):
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = -3
                            record[i] = -4
                            if whetherStl ==1:
                                record[i] = -2
                    else:
                        if TradLen and Pos > TradLen:
                            ClPos[i] = arg_mix[i] + TradLen
                            record[i] = -5
                        else:
                            ClPos[i] = arg_mix[i] + Pos
                            record[i] = 666
                            if whetherStl ==1:
                                record[i] = -2
                elif condiction == 0:
                    continue
                else:
                    arg_mix[i] = 0
                    arg_test[i] = 0
                    error_count += 1
                    print("Error Condiction: ",condiction)
            #Close_test_array
            if tick:
                ClPos_test = ClPos//(12-6*half)
            else:
                ClPos_test = ClPos.copy()
            
            #trigger price record
            Open_price = np.zeros([len(arg_mix),3])
            Close_price = np.zeros([len(ClPos),3])
            for i in range(len(arg_mix)):
                Open_price[i,0] = trigger_stock1[i,(arg_mix[i]+cross+OpenSlipTick)]
                Open_price[i,1] = trigger_stock2[i,(arg_mix[i]+cross+OpenSlipTick)]
                Open_price[i,2] = trigger_spread[i,(arg_mix[i]+cross+OpenSlipTick)]
                Close_price[i,0] = trigger_stock1[i,(ClPos[i]+cross+CloseSlipTick)]
                Close_price[i,1] = trigger_stock2[i,(ClPos[i]+cross+CloseSlipTick)]
                Close_price[i,2] = trigger_spread[i,(ClPos[i]+cross+CloseSlipTick)] 
            
            ForecastPos = -1*np.ones(len(arg_mix))
            ForePrice = np.zeros([len(arg_mix),3])
            if forecast or forelag5:
                
                for i in range(len(arg_test)):
                    if arg_mix[i] != 0:
                        count = 0
                        p,A,ut,_ = fore_chow(test_stock1[i,:150*(1+half)+1],
                                             test_stock2[i,:150*(1+half)+1],
                                             table.model_type.iloc[i],
                                             150*(1+half))
                        for j in range(arg_test[i],TpLen//(1+11*tick)-5):
                            if tick:
                                NowStrL = 150*(1+half)+j+2
                            else:
                                NowStrL = 150*(1+half)+j+cross
                            p,A,ut,StrBr = fore_chow(test_stock1[i,:NowStrL],
                                                     test_stock2[i,:NowStrL],
                                                     table.model_type.iloc[i],
                                                     150*(1+half),True,
                                                     p,A,ut)
                            if forecast:
                                if StrBr == 1:
                                    if tick:
                                        ForecastPos[i] = j+2
                                        NowtickPos = (j+2)*(12-6*half)+CloseSlipTick
                                    else:
                                        ForecastPos[i] = j+cross
                                        NowtickPos = j+cross+CloseSlipTick
                                    ForePrice[i,0] = trigger_stock1[i,NowtickPos]
                                    ForePrice[i,1] = trigger_stock2[i,NowtickPos]
                                    ForePrice[i,2] = trigger_spread[i,NowtickPos]
                                    break
                            else:
                                if StrBr == 1:
                                    count = count + 1
                                    if count == 5:
                                        if tick:
                                            ForecastPos[i] = j+2
                                            NowtickPos = (j+2)*(12-6*half)+CloseSlipTick
                                        else:
                                            ForecastPos[i] = j+cross
                                            NowtickPos = j+cross+CloseSlipTick
                                        ForePrice[i,0] = trigger_stock1[i,NowtickPos]
                                        ForePrice[i,1] = trigger_stock2[i,NowtickPos]
                                        ForePrice[i,2] = trigger_spread[i,NowtickPos]
                                        break
                                else:
                                    count = 0
                            if j == ClPos[i]:
                                break
                            
                            
            table['date'] = ['{}-{}-{}'.format(year,month,day)] * len(table)
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
            Allrecord = NpCombine(Allrecord,record)
            AllForecastPos = NpCombine(AllForecastPos,ForecastPos)
            AllForePrice = NpCombine(AllForePrice,ForePrice)
            

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
record = Allrecord
ForecastPos = AllForecastPos
ForePrice = AllForePrice

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
                ADF_spread = w1 * np.log( test_stock1[i,:(150*(1+half)+arg_test[i])] ) +\
                             w2 * np.log( test_stock2[i,:(150*(1+half)+arg_test[i])] )
            else:   #此處僅為表示tick與非tick算式一樣但想法上有本質上的不同
                ADF_spread = w1 * np.log( test_stock1[i,:(150+arg_test[i])] ) +\
                             w2 * np.log( test_stock2[i,:(150+arg_test[i])] )
            if adfuller( ADF_spread , regression='c' )[1] <= 0.05:
                Open_table_index.append(i)
                int_w.append([w1,w2])
        else:
            Open_table_index.append(i)
            int_w.append([w1,w2])

int_w = np.array(int_w)
Open_table_index = np.array(Open_table_index)

#delete useless table row
table = table.iloc[Open_table_index,:]
arg_test = arg_test[Open_table_index]
arg_mix = arg_mix[Open_table_index]
ClPos = ClPos[Open_table_index]
ClPos_test = ClPos_test[Open_table_index]
LongOrShort = LongOrShort[Open_table_index]
Open_price = Open_price[Open_table_index,:]
Close_price = Close_price[Open_table_index,:]
test_spread = test_spread[Open_table_index,:]
test_stock1 = test_stock1[Open_table_index,:]
test_stock2 = test_stock2[Open_table_index,:]
record = record[Open_table_index]
ForecastPos = ForecastPos[Open_table_index]
ForePrice = ForePrice[Open_table_index]

table = table.reset_index(drop = True)


#Final close position
EndPrice = np.zeros([len(arg_test),3])
EndPos = np.zeros(len(arg_test))
for i in range(len(arg_test)):
    if ForecastPos[i] == -1:
        EndPrice[i,:] = Close_price[i,:]
        EndPos[i] = ClPos_test[i]
    else:
        if ForecastPos[i] <= arg_test[i]:
            EndPrice[i,:] = Close_price[i,:]
            EndPos[i] = ClPos_test[i]
        else:
            EndPrice[i,:] = ForePrice[i,:]
            EndPos[i] = ForecastPos[i]
            record[i] = -3

###Trading###

#Reward
Reward_0 = np.zeros(len(arg_test))
Reward_015 = np.zeros(len(arg_test))
Reward_03 = np.zeros(len(arg_test))
SprReward = np.zeros(len(arg_test))
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
    SprReward[i] = -LongOrShort[i] * Open_price[i,2] + LongOrShort[i] * EndPrice[i,2]


end_time = time.time()
print("Matrix_time:",end_time-start_time)

if check_spread and not half:
    output = np.zeros([2*len(arg_test),274-form_del_min])
    for i in range(len(arg_test)):
        output[2*i,8:258] = test_stock1[i,:]
        output[(2*i+1),8:258] = test_stock2[i,:]
        output[2*i,0] = record[i]
        output[(2*i+1),0] = record[i]
        output[2*i,1] = arg_test[i]
        output[(2*i+1),1] = arg_test[i]
        output[2*i,2] = EndPos[i]
        output[(2*i+1),2] = EndPos[i]
        output[2*i,3] = np.array(table.w1.iloc[i])
        output[(2*i+1),3] = np.array(table.w2.iloc[i])
        output[2*i,4] = int_w[i,0]
        output[(2*i+1),4] = int_w[i,1]
        if Emu:
            output[2*i,5] = np.array(table.Emu.iloc[i])            
            output[(2*i+1),5] = np.array(table.Emu.iloc[i])
        else:
            output[2*i,5] = np.array(table.mu.iloc[i])            
            output[(2*i+1),5] = np.array(table.mu.iloc[i])
        if Estd:
            output[2*i,6] = np.array(table.Estd.iloc[i])
            output[(2*i+1),6] = np.array(table.Estd.iloc[i])
        else:
            output[2*i,6] = np.array(table.stdev.iloc[i])
            output[(2*i+1),6] = np.array(table.stdev.iloc[i])
        output[2*i,7] = Reward_03[i]
        output[(2*i+1),7] = Reward_03[i]
    if save_spread:
        np.save(save_path,output)
if check_spread and half:
    output = np.zeros([2*len(arg_test),519])
    for i in range(len(arg_test)):
        output[2*i,8:] = test_stock1[i,:]
        output[(2*i+1),8:] = test_stock2[i,:]
        output[2*i,0] = record[i]
        output[(2*i+1),0] = record[i]
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
    if save_spread:
        np.save(save_path,output)

print('len:',len(record))
print('Reward_03:',sum(Reward_03))
print('normal close rate:',sum(record==666)/len(record))
print('Profit win rate:',sum(Reward_03>0)/len(Reward_03))


