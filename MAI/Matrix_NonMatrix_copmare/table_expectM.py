# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:15:55 2020

@author: MAI
"""
import numpy as np
import pandas as pd
import os
from vecm import para_vecm
from Matrix_function import order_select
import os

def spread_mean(stock1,stock2,i,table):
    if table.model_type.iloc[i] == 'model1':
        model = 'H2'
    elif table.model_type.iloc[i] == 'model2':
        model = 'H1*'
    elif table.model_type.iloc[i] == 'model3':
        model = 'H1'
    stock1 = stock1[i,:150]
    stock2 = stock2[i,:150]
    b1 = table.w1.iloc[i]
    b2 = table.w2.iloc[i]
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    lyc = logy.copy()
    p = order_select(logy,5)
    #print('p:',p)
    _,_,para = para_vecm(logy,model,p)
    logy = np.mat(logy)
    y_1 = np.mat(logy[p:])
    dy = np.mat(np.diff(logy,axis=0))
    for j in range(len(stock1)-p-1):
        if model == 'H1':
            if p!=1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.hstack([dy[j:(j+p-1)].flatten(),np.mat([1])]).T
            else:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * np.mat([1])
        elif model == 'H1*':
            if p!=1:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T
        elif model == 'H2':
            if p!=1:
                delta = para[0] * para[1].T * y_1[j].T + para[2] * dy[j:(j+p-1)].flatten().T
            else:
                delta = para[0] * para[1].T * y_1[j].T
        else:
            print('Errrrror')
            break
        dy[j+p,:] = delta.T            
        y_1[j+1] = y_1[j] + delta.T
    b = np.mat([[b1],[b2]])
    spread = b.T*lyc[p:].T
    spread_m = np.array(b.T*y_1.T).flatten()
    return spread_m,spread

def get_Estd(stock1,stock2,i,table,dy=True,D=16):
    if table.model_type.iloc[i] == 'model1':
        model = 'H2'
    elif table.model_type.iloc[i] == 'model2':
        model = 'H1*'
    elif table.model_type.iloc[i] == 'model3':
        model = 'H1'
    stock1 = stock1[i,:150]
    stock2 = stock2[i,:150]
    b1 = table.w1.iloc[i]
    b2 = table.w2.iloc[i]
    b = np.mat([[b1],[b2]])
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    u,A,_ = para_vecm(logy,model,p)
    constant = np.mat(A[:,0])
    A = A[:,1:]
    l = A.shape[1]
    extend = np.hstack([np.identity(l-2),np.zeros([l-2,2])])
    newA = np.vstack([A,extend])
    if not dy:
        lagy = logy[p-1:-1,:]
        for i in range(1,p):
            lagy = np.hstack([lagy,logy[p-1-i:-i-1,:]])
        MatrixA = np.mat(A)
        MatrixLagy = np.mat(lagy)
        Estimate_logy = MatrixA * MatrixLagy.T + constant
        e = logy[p:,:].T-Estimate_logy
        var = e*e.T/e.shape[1]
    else:
        var = u*u.T/u.shape[1]
    NowCoef = np.mat(np.eye(len(newA)))
    Evar = var.copy()
    for i in range(149):
        NowCoef = newA * NowCoef
        Evar = Evar + NowCoef[:2,:2]*var*NowCoef[:2,:2].T
    Evar = b.T * Evar * b
    
    return np.sqrt(Evar)

save_path = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200619Etestcorrect'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\Origin_table'
tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200619testcorrect'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\allow_R2_table'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20200204\forelag5_min_open0.85_close0.75'
#tablePath = r'C:\Users\MAI\Desktop\trading_result\after20191128\cnn_min_open0.85_close0.75_new'
y = 2018
#tablePath = r'C:\Users\MAI\Desktop\trading_result\fore_lag5_half_thres0.45_open0.85close_0.85'
test_dataPath = r'C:\Users\MAI\Desktop\{}\averageprice'.format(y)
#test_dataPath = r'C:\Users\MAI\Desktop\half-min\2016\0050'
test_volumePath = r'C:\Users\MAI\Desktop\min-API\accumulate_volume'

years = ["{}".format(y)]
#months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
#months = ["07","08","09","10","11","12"]
months = ["02"]
#days = ["01","02","03","04","05","06","07","08","09","10",
#       "11","12","13","14","15","16","17","18","19","20",
#       "21","22","23","24","25","26","27","28","29","30",
#       "31"]
days = ["05"]
#Clock
form_del_min = 16
test_csv_name = '_averagePrice_min'

for year in years:
    for month in months:
        print("Now import: ",month,"-th month")
        for day in days:
            try:
                #Read data
#                if year == '2018':
#                    table = pd.read_csv(os.path.join(tablePath,'{}{}{}.csv'.format(year,month,day)))
#                else:
                table = pd.read_csv(os.path.join(tablePath,'{}{}{}_table.csv'.format(year,month,day)))
                test_data = pd.read_csv(os.path.join(test_dataPath,'{}{}{}{}.csv'.format(year,month,day,test_csv_name)))
                test_data = test_data.iloc[form_del_min:,:]
                test_data = test_data.reset_index(drop=True)
            except:
                continue
            stock1_name = table.stock1.astype('str',copy=False)
            stock2_name = table.stock2.astype('str',copy=False)
            test_stock1 = np.array(test_data[stock1_name].T)
            test_stock2 = np.array(test_data[stock2_name].T)
            mean = np.zeros(len(table))
            std = np.zeros(len(table))
            for i in range(len(table)):
                spread_m,spread = spread_mean(test_stock1,test_stock2,i,table)
                mean[i] = np.mean(spread_m[-1:])
                #std[i] = np.sqrt(np.mean(np.square(spread_m-spread)))
                std[i] = get_Estd(test_stock1,test_stock2,i,table)
            table['Emu'] = mean
            table['Estd'] = std
            table.to_csv(os.path.join(save_path,'{}{}{}_table.csv'.format(year,month,day)),index=False)