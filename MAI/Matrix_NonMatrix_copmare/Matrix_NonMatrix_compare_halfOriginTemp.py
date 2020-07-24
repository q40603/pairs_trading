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
from keras.models import load_model
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

def CNN_test(st1,st2,sp,tick,DetPos,table,NowOpen):
    if NowOpen:
        times = 1
    else:
        times = len(DetPos)
    #Array Initialize
    AllSprInput = []
    AllFSprInput = []
    AllCharInput = []
    pair_pos = np.zeros([len(DetPos)],dtype = int)
    count = 0
    
    character = ['w1','w2','mu','stdev']
    TableChar = np.zeros([len(table),4])
    TableChar[:,:4] = np.array(table[character])

    if tick:
        s = 200
    else:
        s = 200
    for m in range(times):
        SprInput = np.zeros([100,3])
        CharInput = np.zeros([4])
        if not NowOpen:
            CharInput[:4] = TableChar[m,:4]
            lenth = len(DetPos[m])
        else:
            lenth = len(DetPos)
        for i in range(lenth):
            if NowOpen:
                index = DetPos[i]
                pair = i
            else:
                index = DetPos[m][i,0]
                pair = m
            SprInput[:,0] = st1[pair,(s+index):(s+100+index)]
            SprInput[:,1] = st2[pair,(s+index):(s+100+index)]
            SprInput[:,2] = sp[pair,(s+index):(s+100+index)]

            AllSprInput.append(SprInput.copy())
            if NowOpen:
                CharInput[:4] = TableChar[i,:4]
            AllCharInput.append(CharInput.copy())
            count += 1
        pair_pos[m] = count
    AllSprInput = np.array(AllSprInput)
    AllCharInput = np.array(AllCharInput)
    #Normalize CNN_SpreadInput
    #mu
    mu = np.zeros([len(AllSprInput),1,3])
    mu[:,0,:2] = np.mean(AllSprInput[:,:,:2], axis=1)
    mu[:,0,2] = AllCharInput[:,2]
    #std
    stock_std = np.std(AllSprInput[:,:,:3], axis=1)
    std = np.ones([len(AllSprInput),1,3])
    std[:,0,:2] = stock_std[:,:2]
    std[:,0,2] = AllCharInput[:,3]
    #Normalize
    AllSprInput = (AllSprInput - mu)/std
    AllCharInput[:,:2] = AllCharInput[:,:2]*stock_std[:,:2] / np.expand_dims(stock_std[:,2],axis = 1)
        
    #CNN_predict
    pre = model_CNN.predict([AllSprInput,AllCharInput])
    prediction = np.argmax(pre,axis = 1)
    
    if NowOpen:
        return prediction
    else:
        return [ prediction , pair_pos ]
#initialize
tick = True                # using tick data or not
half = True
EarlyStop = 0*(1+half)              # After how many minutes, we stop opening spread 
TradLen = 0*(1+tick)               # Trading lenth
Cost_threshold = True       # whether to use cost threshold
Adf = False                  # Whether to use Adf-check before open
CNN_open =True             # Whether to use CNN-check before open
CNN_det = True              # Whether to use CNN-check during trading period
open_thres = 0.85           # Open threshold (std)
close_thres = 0.85          # Close threshold (std)
stoploss_thres = 0
cost_gate = 0.0045           # cost gate threshold (percent)
form_del_min = 16           # how many minutes are deleted in formation period
capital = 5000              # capital
maxi = 5                    # restrict the number of ticket we could buy for a stock
OpenSlipTick = 0            # slip how many tick when open
CloseSlipTick = 1           # slip how many tick when close
TpLen = 100 * ( 1 + tick )
save_spread = False
save_path = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\result\O{}C{}E{}.npy'.format(open_thres,close_thres,EarlyStop)
MCheck = 0

model_name = 'model_combine_compare_200times'
model_CNN = load_model(r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\model\{}.h5'.format(model_name))
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table'
tablePath = r'C:\Users\MAI\Desktop\trading_result\after20191128\cnn_half_compare_open0.85_close0.85'
#test_dataPath = r'C:\Users\MAI\Desktop\2016\averageprice'
test_dataPath = r'C:\Users\MAI\Desktop\half-min\2016\0050'
#test_volumePath = r'C:\Users\MAI\Desktop\min-API\accumulate_volume'
test_volumePath = r'C:\Users\MAI\Desktop\min-API\accumulate_volume'
tick_trigger_dataPath = r'D:\tick(secs)\2016'
origin_trigger_dataPath = r'C:\Users\MAI\Desktop\2016\minprice'
if half:
    test_csv_name = '_half_min'
else:
    test_csv_name = '_averagePrice_min'


years = ["2016"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
#months = ["03"]
days = ["01","02","03","04","05","06","07","08","09","10",
       "11","12","13","14","15","16","17","18","19","20",
       "21","22","23","24","25","26","27","28","29","30",
       "31"]
#days = ["21"]
#Clock
start_time = time.time()
for year in years:
    for month in months:
        print("Now import: ",month,"-th month")
        for day in days:
            try:
                #Read data
                table = pd.read_csv(os.path.join(tablePath,'{}{}{}_table.csv'.format(year,month,day)))
                test_data = pd.read_csv(os.path.join(test_dataPath,'{}{}{}{}.csv'.format(year,month,day,test_csv_name)))
                test_data = test_data.iloc[(form_del_min*(1+half)):,:]
                test_data = test_data.reset_index(drop=True)
                if tick:
                    trigger_data = pd.read_csv(os.path.join(tick_trigger_dataPath,'{}{}{}_tick_stock.csv'.format(year,month,day)))
                    sec_5 = np.arange(8975+60*form_del_min,trigger_data.shape[0]-1,30)
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
            if len(table) == 0:
                continue
            #Preprocessing data --> VECM spread
            stock1_name = table.stock1.astype('str',copy=False)
            stock2_name = table.stock2.astype('str',copy=False)
            drop = []
            for i in range(len(stock1_name)):
                q1 = stock1_name.iloc[i] not in trigger_data.columns
                q2 = stock2_name.iloc[i] not in trigger_data.columns
                if q1 or q2:
                    drop.append(i)
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
                trigger_stock1 = trigger_stock1[:,:-9]
                trigger_stock2 = trigger_stock2[:,:-9]
                trigger_spread = trigger_spread[:,:-9]
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
            CNN_up = np.expand_dims(mu+std,axis = 1)
            CNN_down = np.expand_dims(mu-std,axis = 1)
            Stl_up = np.expand_dims(mu+stoploss_thres*std,axis = 1)
            Stl_down = np.expand_dims(mu-stoploss_thres*std,axis = 1)
            
            #Where_cross_threshold
            OpCheck_up = Where_cross_threshold(trigger_spread, up_open, 1)
            OpCheck_down = Where_cross_threshold(trigger_spread, down_open, 3)
            ClCheck_up = Where_cross_threshold(trigger_spread, up_close, 1)
            ClCheck_down = Where_cross_threshold(trigger_spread, down_close, 3)
            CNNCheck_up = Where_cross_threshold(trigger_spread, CNN_up, 1)
            CNNCheck_down = Where_cross_threshold(trigger_spread, CNN_down, 3)
            StlCheck_up = Where_cross_threshold(trigger_spread, Stl_up, 1)
            StlCheck_down = Where_cross_threshold(trigger_spread, Stl_down, 3)
            
            #Combine open trigger array
            OpMix = OpCheck_up+OpCheck_down
            arg_mix = np.argmax(OpMix!=0,axis = 1)
            
            #Open_test_array
            if tick:
                arg_test = arg_mix.copy()
            else:
                arg_test = arg_mix.copy()
            
            if MCheck:
                M = 1//np.array(table.iloc[:,7])
                for i in range(len(arg_mix)):
                    if arg_test[i]>M[i]:
                        arg_mix[i] = 0
                        arg_test[i] = 0
                        
            #EarlyStop
            if EarlyStop:
                arg_mix[arg_test>EarlyStop] = 0
                arg_test[arg_test>EarlyStop] = 0
            
            
            #if not open, return "-1"
            ClPos = np.ones(len(arg_mix))*(-7)
            ClPos = np.int64(ClPos)
            CNNDetPos = []
            CNNDetPos_test = []
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
                        if S!= 0:
                            if S<Pos or Pos==0:
                                Pos = S
                                whetherStl = 1
                    if Pos == 0 or (arg_mix[i]+Pos) > (TpLen-3):
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
                    whereDet = np.argwhere(CNNCheck_up[i,arg_mix[i]:ClPos[i]]>0)
                    CNNDetPos.append( (arg_mix[i] + whereDet) )
                    if tick:
                        CNNDetPos_test.append( (arg_mix[i] + whereDet) )
                    else:
                        CNNDetPos_test.append( (arg_mix[i] + whereDet) )
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
                    whereDet = np.argwhere(CNNCheck_down[i,arg_mix[i]:ClPos[i]]<0)
                    CNNDetPos.append( (arg_mix[i] + whereDet) )
                    if tick:
                        CNNDetPos_test.append( (arg_mix[i] + whereDet) )
                    else:
                        CNNDetPos_test.append( (arg_mix[i] + whereDet) )
                elif condiction == 0:
                    CNNDetPos.append(np.array([]))
                    CNNDetPos_test.append(np.array([]))
                else:
                    print("Error Condiction: ",condiction)
            #Close_test_array
            if tick:
                ClPos_test = ClPos.copy()
            else:
                ClPos_test = ClPos.copy()
            
            #trigger price record
            Open_price = np.zeros([len(arg_mix),3])
            Close_price = np.zeros([len(ClPos),3])
            CNNBreak_price = []
            for i in range(len(arg_mix)):
                Open_price[i,0] = trigger_stock1[i,(arg_mix[i]+1+OpenSlipTick)]
                Open_price[i,1] = trigger_stock2[i,(arg_mix[i]+1+OpenSlipTick)]
                Open_price[i,2] = trigger_spread[i,(arg_mix[i]+1+OpenSlipTick)]
                Close_price[i,0] = trigger_stock1[i,(ClPos[i]+1+CloseSlipTick)]
                Close_price[i,1] = trigger_stock2[i,(ClPos[i]+1+CloseSlipTick)]
                Close_price[i,2] = trigger_spread[i,(ClPos[i]+1+CloseSlipTick)] 
                if len(CNNDetPos[i])!=0:
                    CNNBreak_price.append( [trigger_stock1[i,(CNNDetPos[i]+1+CloseSlipTick)],
                                            trigger_stock2[i,(CNNDetPos[i]+1+CloseSlipTick)],
                                            trigger_spread[i,(CNNDetPos[i]+1+CloseSlipTick)]] )
                else:
                    CNNBreak_price.append([])
            
            table['date'] = ['{}{}{}'.format(year,month,day)] * len(table)
            Alltable = PdCombine(Alltable,table)
            Allarg_test = NpCombine(Allarg_test,arg_test)
            Allarg_mix = NpCombine(Allarg_mix,arg_mix)
            AllClPos = NpCombine(AllClPos,ClPos)
            AllClPos_test = NpCombine(AllClPos_test,ClPos_test)
            AllCNNDetPos = AllCNNDetPos + CNNDetPos
            AllCNNDetPos_test = AllCNNDetPos_test + CNNDetPos_test
            AllLongOrShort = NpCombine(AllLongOrShort,LongOrShort)
            AllOpen_price = NpCombine(AllOpen_price,Open_price)
            AllClose_price = NpCombine(AllClose_price,Close_price)
            AllCNNBreak_price = AllCNNBreak_price + CNNBreak_price
            Alltest_spread = NpCombine(Alltest_spread,test_spread)
            Alltest_stock1 = NpCombine(Alltest_stock1,test_stock1)
            Alltest_stock2 = NpCombine(Alltest_stock2,test_stock2)
            Allrecord = NpCombine(Allrecord,record)

#change
table = Alltable
arg_test = Allarg_test
arg_mix = Allarg_mix
ClPos = AllClPos
ClPos_test = AllClPos_test
CNNDetPos = AllCNNDetPos
CNNDetPos_test = AllCNNDetPos_test
LongOrShort = AllLongOrShort
Open_price = AllOpen_price
Close_price = AllClose_price
CNNBreak_price = AllCNNBreak_price
test_spread = Alltest_spread
test_stock1 = Alltest_stock1
test_stock2 = Alltest_stock2
record = Allrecord

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
                ADF_spread = w1 * np.log( test_stock1[i,:(150*(1*half)+arg_test[i])] ) +\
                             w2 * np.log( test_stock2[i,:(150*(1*half)+arg_test[i])] )
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
CNNDetPos = [CNNDetPos[i] for i in Open_table_index]
CNNDetPos_test = [CNNDetPos_test[i] for i in Open_table_index]
LongOrShort = LongOrShort[Open_table_index]
Open_price = Open_price[Open_table_index,:]
Close_price = Close_price[Open_table_index,:]
CNNBreak_price = [CNNBreak_price[i] for i in Open_table_index]
test_spread = test_spread[Open_table_index,:]
test_stock1 = test_stock1[Open_table_index,:]
test_stock2 = test_stock2[Open_table_index,:]
record = record[Open_table_index]


#CNN Test
if CNN_open:
    prediction = CNN_test(test_stock1, test_stock2, test_spread, tick, 
                          arg_test, table, True)
    #Clean Open Table and Position
    Open_table_index = np.arange(len(prediction))[prediction==1]
    table = table.iloc[Open_table_index,:]
    arg_test = arg_test[Open_table_index]
    arg_mix = arg_mix[Open_table_index]
    ClPos = ClPos[Open_table_index]
    ClPos_test = ClPos_test[Open_table_index]
    CNNDetPos = [CNNDetPos[i] for i in Open_table_index]
    CNNDetPos_test = [CNNDetPos_test[i] for i in Open_table_index]
    LongOrShort = LongOrShort[Open_table_index]
    Open_price = Open_price[Open_table_index,:]
    Close_price = Close_price[Open_table_index,:]
    CNNBreak_price = [CNNBreak_price[i] for i in Open_table_index]
    int_w = int_w[Open_table_index]
    test_spread = test_spread[Open_table_index,:]
    test_stock1 = test_stock1[Open_table_index,:]
    test_stock2 = test_stock2[Open_table_index,:]
    record = record[Open_table_index]
    #CNN_SpreadInput = CNN_SpreadInput[Open_table_index]

table = table.reset_index(drop = True)

#Stop Loss position
if CNN_det:
    prediction, pair_pos = CNN_test(test_stock1, test_stock2, test_spread, 
                                    tick,CNNDetPos_test, table, False)
else:
    prediction = np.ones(len(arg_test))
    pair_pos = np.arange(len(arg_test))

#Final close position
EndPrice = np.zeros([len(arg_test),3])
breakPoint = []
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
        index = np.argmax(prediction[last:pair_pos[i]]==0)
        EndPrice[i,0] = CNNBreak_price[i][0][index,0]
        EndPrice[i,1] = CNNBreak_price[i][1][index,0]
        EndPrice[i,2] = CNNBreak_price[i][2][index,0]
        record[i] = -3
        breakPoint.append(CNNDetPos_test[i][index,0])
    else:
        print("Close condiction error.")
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

if save_spread:
    output = np.zeros([2*len(arg_test),508])
    for i in range(len(arg_test)):
        output[2*i,8:258] = test_stock1[i,:]
        output[(2*i+1),8:258] = test_stock2[i,:]
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
    np.save(save_path,output)