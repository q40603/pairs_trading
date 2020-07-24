# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:50:23 2020

@author: MAI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import statsmodels.api as sm
from statsmodels.tsa.api import AutoReg
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from vecm import para_vecm
from Matrix_function import order_select
from statsmodels.tsa.arima_model import ARIMA
import scipy.stats as stats
import math
from collections import defaultdict
from datetime import datetime
import matplotlib.dates as mdates

def standard(x):
    x = (x-np.mean(x))/np.std(x)
    return x
def plotSprV1V2(i,check):
    plt.figure(figsize=(12.5,10))
    spread = check[2*i,3]*np.log(check[2*i,8:258]) + check[2*i+1,3]*np.log(check[2*i+1,8:258])
    mu = check[2*i,5]
    std = check[2*i,6]
    plt.subplot(5,1,1)
    plt.plot(spread)
    plt.axhline(mu)
    plt.axhline(mu+1.5*std,color='r')
    plt.axhline(mu-1.5*std,color='r')
    plt.axvline(150+check[2*i,1],color='black')
    if check[2*i,2] != -1:
        plt.axvline(150+check[2*i,2],color='black')
    else:
        plt.axvline(249,color='black')
        
    plt.subplot(5,1,2)
    plt.plot(check[2*i,8:258])
    plt.subplot(5,1,3)
    plt.plot(check[2*i+1,8:258])
    plt.subplot(5,1,4)
    plt.plot(check[2*i,258:508])
    plt.subplot(5,1,5)
    plt.plot(check[2*i+1,258:508])
    print('Position:',check[2*i,0])


def Where_cross_threshold(trigger_spread, threshold, add_num):
    #initialize array
    check = np.zeros(trigger_spread.shape)
    #put on the condiction
    check[(trigger_spread - threshold) > 0] = add_num
    check[:,0] = check[:,1]
    #Open_trigger_array
    check = check[:,1:] - check[:,:-1]
    return check

def zero(check):
    spread = np.zeros([len(check)//2,250])
    mu = np.zeros([len(check)//2,1])
    for i in range(len(check)//2):
        spread[i,:] = check[2*i,3]*np.log(check[2*i,8:258]) + check[2*i+1,3]*np.log(check[2*i+1,8:258])
        mu[i,0] = check[2*i,5]        
    where = Where_cross_threshold(spread[:,8:158],mu,1)
    count = np.sum(where!=0,axis = 1)
    return count

def stdCheck(check):
    spread = np.zeros([len(check)//2,250])
    mu = np.zeros([len(check)//2,1])
    for i in range(len(check)//2):
        spread[i,:] = check[2*i,3]*np.log(check[2*i,8:258]) + check[2*i+1,3]*np.log(check[2*i+1,8:258])
        mu[i,0] = check[2*i,5]        
    sqr1 = np.sum((spread[:,:75]-mu)**2,axis=1)
    sqr2 = np.sum((spread[:,75:150]-mu)**2,axis=1)
    rate = sqr2/sqr1
    return rate

def areaCheck(check):
    spread = np.zeros([len(check)//2,250])
    mu = np.zeros([len(check)//2,1])
    std = np.zeros([len(check)//2,1])
    for i in range(len(check)//2):
        spread[i,:] = check[2*i,3]*np.log(check[2*i,8:258]) + check[2*i+1,3]*np.log(check[2*i+1,8:258])
        mu[i,0] = check[2*i,5]        
        std[i,0] = check[2*i,6]
    area = np.abs(np.sum((spread[:,:150]-mu)/std,axis=1))
    return area
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values
def fftCheck(check,where):
    old = []
    new = []
    Dif = []
    SDif = []
    for i in range(len(check)//2):
        spread = check[2*i,3]*np.log(check[2*i,8:258]) + check[2*i+1,3]*np.log(check[2*i+1,8:258])
        spread = standard(spread[:150])
        x1,y1 = get_fft_values(spread[25:100],1/25,75,25)
        x2,y2 = get_fft_values(spread[75:150],1/25,75,25)

#        x2,y2 = get_fft_values(spread[int(check[2*i,1]+50):(int(check[2*i,1])+150)],1/25,100,25)
        old.append(sum(y1[:where]))
        new.append(sum(y2[:where]))
        Dif.append(sum(np.square(y1[:where]-y2[:where])))
        SDif.append(sum(np.square(y1[:where]-y2[:where])))
    return old,new,Dif,SDif

def fftHalfCheck(check,where):
    old = []
    new = []
    Dif = []
    for i in range(len(check)//2):
        spread = check[2*i,3]*np.log(check[2*i,8:]) + check[2*i+1,3]*np.log(check[2*i+1,8:])
        spread = standard(spread[:300])
        x1,y1 = get_fft_values(spread[50:300],1/25,200,25)
        x2,y2 = get_fft_values(spread[int(check[2*i,1]+100):(int(check[2*i,1])+300)],1/25,200,25)
        old.append(sum(y1[:where]))
        new.append(sum(y2[:where]))
        Dif.append(sum(y1[:where]-y2[:where]))
    return old,new,Dif

def fftTraf(check,i,D):
    spread = check[2*i,3]*np.log(check[2*i,8:(274-D)]) + check[2*i+1,3]*np.log(check[2*i+1,8:(274-D)])

    plt.figure(figsize=(16,4))
    mu = check[2*i,5]
    std = check[2*i,6]
#    plt.subplot(3,1,1)
    plt.plot(spread)
    plt.axhline(mu)
    plt.axhline(mu+1.5*std,color='r')
    plt.axhline(mu-1.5*std,color='r')
    plt.axvline(150+check[2*i,1],color='black')
    if check[2*i,2] == -3 or check[2*i,2] == -1:
        plt.axvline(266-D,color='black')
    else:
        plt.axvline(150+check[2*i,2],color='black')
    print('Position:',check[2*i,0])
#    spread1 = standard(spread[:(166-D)])
##    x1,y1 = get_fft_values(spread[25:150],1/25,100,25)
##    x2,y2 = get_fft_values(spread[int(check[2*i,1]+50):(int(check[2*i,1])+150)],1/25,100,25)
#    x1,y1 = get_fft_values(spread1[25:100],1/25,75,25)
#    x2,y2 = get_fft_values(spread1[75:150],1/25,75,25)
#    plt.subplot(3,1,2)
#    plt.plot(x1,y1)
#    plt.subplot(3,1,3)
#    plt.plot(x2,y2)
#    print('y1:',sum(y1[:4]))
#    print('y2:',sum(y2[:4]))
#    print('Dif:',sum(y1[:4]-y2[:4]))
#    print('SDif:',sum(np.square(y1[:4]-y2[:4])))
#    print('mu:',mu)
#    print('Actual mu:',np.mean(spread[:150]))
def get_IBA(output,i,model,D=16,pri=False,Ori=False):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    #print('p:',p)
    #print('model:',model)
    _,_,para = para_vecm(logy,model,p)
    if model == 'H1*':
        para[1] = para[1][:2,:] 
    H = np.eye(para[1].shape[1])+para[1].T*para[0]
    H2 = np.eye(para[0].shape[0])+para[0]*para[1].T
    eva,evct = np.linalg.eig(H)
    eva2,evct2 = np.linalg.eig(H2)
    if pri == True:
        print('H:',H)
        print('B\'A+I:',eva)
        print('AB\'+I:',eva2)
    if Ori==False:
        return eva,evct
    else:
        return eva,evct,eva2

def spread_mean(output,i,model,D,ini=0,future=True,order=False):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    b1 = output[2*i,3]
    b2 = output[(2*i+1),3]
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    #print('p:',p)
    _,_,para = para_vecm(logy,model,p)
    logy = np.mat(logy)
    y_1 = np.mat(logy[p+ini:])
    dy = np.mat(np.diff(logy,axis=0)[ini:,:])
    for j in range(len(stock1)-p-1-ini):
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
    spread_m = np.array(b.T*y_1.T).flatten()
    if future:
        if order:
            return spread_m,spread[p+ini:],p
        else:
            return spread_m,spread[p+ini:]
    else:
        if order:
            return spread_m,spread[p+ini:150],p
        else:
            return spread_m,spread[p+ini:150]

    
def ex_spread_mean(output,i,model,D,l,ini=0):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    b1 = output[2*i,3]
    b2 = output[(2*i+1),3]
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    #print('p:',p)
    _,_,para = para_vecm(logy,model,p)
    stock1 = output[2*i,8:]
    stock2 = output[(2*i+1),8:]
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    logy = np.mat(logy)
    y_1 = np.mat(np.zeros([l-p-ini,2]))
    y_1[0,:] = np.mat(logy[p+ini,:])
    dy = np.mat(np.zeros([l-ini,2]))
    dy[:(p-1),:] = np.mat(np.diff(logy,axis=0)[ini:(ini+p-1),:])
    for j in range(l-p-1-ini):
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
    spread_m = np.array(b.T*y_1.T).flatten()
    return spread_m,spread[p+ini:]

def get_root(output,i,model,D):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    _,A,_ = para_vecm(logy,model,p)
    A = A[:,1:]
    l = A.shape[1]
    extend = np.hstack([np.identity(l-2),np.zeros([l-2,2])])
    newA = np.vstack([A,extend])
    eigv = np.linalg.eig(newA)[0]
    absEv = np.abs(eigv)
    explode = 0
    unit_root = 1
    unit_root_i = 0
    for i in range(l):
        if absEv[i]>1.001:
            explode = 1
        if absEv[i]>0.999:
            if np.iscomplex(eigv[i]):
                unit_root_i = 1
            else:
                unit_root = 0
    
    #explode = 0
    unit_root = 0
    #unit_root_i = 0

    condiction = explode or unit_root or unit_root_i
    return eigv,condiction

def get_Estd(output,i,model,dy=True,D=16,iteration=150):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    b1 = output[2*i,3]
    b2 = output[(2*i+1),3]
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
    for i in range(iteration-1):
        NowCoef = newA * NowCoef
        Evar = Evar + NowCoef[:2,:2]*var*NowCoef[:2,:2].T
    Evar = b.T * Evar * b
    
    return np.sqrt(Evar)

def spread_coef(output,i,model,D):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    b1 = output[2*i,3]
    b2 = output[(2*i+1),3]
    b = np.mat([[b1],[b2]])
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    _,coef,_ = para_vecm(logy,model,p)
    ARcoef = b.T * coef
    return ARcoef
    
def spread_mean_plot(output,i,model,D,ini=0,future=True):
    mu = output[2*i,5]
    std = output[2*i,6]
    spread_m,spread = spread_mean(output,i,model,D,ini,future)
    plt.figure(figsize=(16,4))
    plt.axhline(mu,color='black',label='mu')
    plt.axhline(mu+1.5*std,color='r',label='open_thres')
    plt.axhline(mu-1.5*std,color='r')
    plt.plot(spread,label='spread')
    plt.plot(spread_m,label='x_hat')
    plt.title('num:{},record:{}'.format(i,output[2*i,0]))
    plt.legend()

def only_mean_plot(output,i,model,l,D,s_expand = False):
    if model == 'model1':
        model = 'H2'
    elif model == 'model2':
        model = 'H1*'
    elif model == 'model3':
        model = 'H1'
    stock1 = output[2*i,8:158]
    stock2 = output[(2*i+1),8:158]
    b1 = output[2*i,3]
    b2 = output[(2*i+1),3]
    y = np.vstack( [stock1, stock2] ).T
    logy = np.log(y)
    p = order_select(logy,5)
    #print('p:',p)
    _,_,para = para_vecm(logy,model,p)
    logy = np.mat(logy)
    y_1 = np.mat(np.zeros([l-p,2]))
    y_1[0,:] = np.mat(logy[p,:])
    dy = np.mat(np.zeros([l,2]))
    dy[:(p-1),:] = np.mat(np.diff(logy,axis=0)[:(p-1),:])
    for j in range(l-p-1):
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
        dy[j+p,:]=delta.T
        y_1[j+1] = y_1[j] + delta.T
    b = np.mat([[b1],[b2]])
    spread_m = np.array(b.T*y_1.T).flatten()
    
    mu = output[2*i,5]
    std = output[2*i,6]
    
    if s_expand == True:
        y = np.vstack( [stock1, stock2] ).T
        logy = np.log(y)
        logy = np.mat(logy)
        y_1 = np.mat(np.zeros([l-p,2]))
        y_1[:150-p,:] = np.mat(logy[p:,:])
        dy = np.mat(np.zeros([l,2]))
        dy[:149,:] = np.mat(np.diff(logy,axis=0))
        for j in range(len(logy)-p-2,l-p-1):
            if model == 'H1':
                if p!=1:
                    delta = para[0] * para[1].T * y_1[j].T + para[2] * np.hstack([dy[(j+1):(j+p)].flatten(),np.mat([1])]).T + std* np.mat(np.random.rand(2)).T
                else:
                    delta = para[0] * para[1].T * y_1[j].T + para[2] * np.mat([1]) + std * np.mat(np.random.rand(2)).T
            elif model == 'H1*':
                if p!=1:
                    delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T + para[2] * dy[(j+1):(j+p)].flatten().T + std * np.mat(np.random.rand(2)).T
                else:
                    delta = para[0] * para[1].T * np.hstack([y_1[j],np.mat([1])]).T + std * np.mat(np.random.rand(2)).T
            elif model == 'H2':
                if p!=1:
                    delta = para[0] * para[1].T * y_1[j].T + para[2] * dy[(j+1):(j+p)].flatten().T + std * np.mat(np.random.rand(2)).T
                else:
                    delta = para[0] * para[1].T * y_1[j].T + np.mat(np.random.rand(2)).T + std * np.mat(np.random.rand(2)).T
            else:
                print('Errrrror')
                break          
            dy[j+p,:]=delta.T
            y_1[j+1] = y_1[j] + delta.T
        spread = np.array(b.T*y_1.T).flatten()
        
    else:
        spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
        spread = spread[p:]
    
    plt.figure(figsize=(16,4))
    plt.axhline(mu)
    plt.axhline(mu+1.5*std,color='r')
    plt.axhline(mu-1.5*std,color='r')
    plt.plot(spread)
    plt.plot(spread_m,color = 'orange')

def BS_check(output,i,table,formation = True):
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    if formation:
        stock1 = output[2*i,8:158]
        stock2 = output[(2*i+1),8:158]
    else:
        stock1 = output[2*i,8:]
        stock2 = output[(2*i+1),8:]
    logS1 = np.log(stock1)
    logS2 = np.log(stock2)
    dflS1 = np.diff(logS1)
    dflS2 = np.diff(logS2)
    standR1 = standard(dflS1)
    standR2 = standard(dflS2)
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.title('stock:{},beta:{}'.format(table.iloc[i,0],output[2*i,3]))
    plt.hist(standR1,density = True)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.subplot(1,2,2)
    plt.title('stock:{},beta:{}'.format(table.iloc[i,1],output[(2*i+1),3]))
    plt.hist(standR2,density = True)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

def BS_5check(output,start,table,SprReward_03,sort_order,formation = True):   
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.figure(figsize=(9,45))
    plt.subplots_adjust(wspace =0.5, hspace =0.5)
    for i in range(start,start+5):
        if formation:
            stock1 = output[2*i,8:158]
            stock2 = output[(2*i+1),8:158]
        else:
            stock1 = output[2*i,8:]
            stock2 = output[(2*i+1),8:]
        logS1 = np.log(stock1)
        logS2 = np.log(stock2)
        dflS1 = np.diff(logS1)
        dflS2 = np.diff(logS2)
        standR1 = standard(dflS1)
        standR2 = standard(dflS2)
        #plt.figure(figsize=(16,4))
        plt.subplot(20,2,2*i+1-start)
        plt.title('stock:{},beta:{}'.format(table.iloc[i,0],output[2*i,3]))
        plt.hist(standR1,density = True)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.subplot(20,2,2*i+2-start)
        plt.title('stock:{},beta:{}'.format(table.iloc[i,1],output[(2*i+1),3]))
        plt.hist(standR2,density = True)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        print('{}th return:'.format(i+1),SprReward_03[sort_order[i]])

def threshold_plot(array,num):
    totalP = sum(array[:,0] == 666)
    totalF = sum(array[:,0] != 666)
    maximum = max(array[:,1])
    minimum = min(array[:,1])
    step = (maximum - minimum)/num
    threshold = np.zeros(num)
    Positive = np.zeros(num)
    Negative = np.zeros(num)
    for i in range(num):
        threshold[i] = minimum + step*(i+1)
        #Positive[i] = sum(array[array[:,1]>threshold[i],0]==666)/totalP
        Positive[i] = totalP-sum(array[array[:,1]>threshold[i],0]==666)
        #Negative[i] = sum(array[array[:,1]>threshold[i],0]!=666)/totalF
        Negative[i] = totalF-sum(array[array[:,1]>threshold[i],0]!=666)
    #plt.plot(threshold,Positive,label = 'Positive Remain rate')
    #plt.plot(threshold,Negative,label = 'Negative Remain rate')
    plt.plot(threshold,Positive,label = 'Positive Lost Num')
    plt.plot(threshold,Negative,label = 'Negative Lost Num')
    plt.legend()
    return threshold,Positive,Negative

def MarkovSwitching(output,i,D):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = sm.tsa.MarkovAutoregression(spread1, k_regimes=2, order=3, 
                                        switching_ar=False,
                                        switching_trend=True,
                                        switching_exog=False,
                                        switching_variance=False)
    fit = model.fit()
    print('markov:',fit.bic)
    model2 = AutoReg(spread1,lags = 1)
    fit2 = model2.fit()
    print('AR:',fit2.bic)
#    print(fit.summary())
#    print(fit.smoothed_marginal_probabilities)
#    plt.fill_between(range(len(output)),0,1,color = 'k',alpha = 0.1)

def MarkovCheck(output,i,D):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = sm.tsa.MarkovAutoregression(spread1, k_regimes=2, order=4, switching_ar=False)
    fit = model.fit()
    model2 = AutoReg(spread1,lags = 4)
    fit2 = model2.fit()
    if fit.aic>fit2.aic:
        return 1
    else:
        return 0
def MarkovSave(output,i,D,order):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = sm.tsa.MarkovAutoregression(spread1, k_regimes=2, order=order, switching_ar=False)
    fit = model.fit()
    return fit.bic

def save_to_csv(output,i,D,good):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    spread1 = pd.DataFrame(spread1)
    spread1.columns = ['price']
    if good==True:
        spread1.to_csv('good{}.csv'.format(i))
    else:
        spread1.to_csv('bad{}.csv'.format(i))

def ARSave(output,i,D,order):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = AR(spread1)
    fit = model.fit(order)
    return fit.bic

def AutoRegSave(output,i,D,order):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = AutoReg(spread1,lags = order)
    fit = model.fit()
    return fit.bic

def MyBicSave(output,i,D,order,dyn=False):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = AutoReg(spread1,lags = order)
    fit = model.fit()
    pre = fit.predict(start = order,end = (len(spread1)-1),dynamic = dyn)
    n = len(spread1)
    mse = mean_squared_error(spread1[order:],pre)
    bic = (order+1)*np.log(n)/n + np.log(n*mse/(n-order))
    return bic

def ARIMASave(output,i,D,order):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = ARIMA(spread1,(order,0,0))
    fit = model.fit(method='css')
    return fit.bic

def ARplot(output,i,D,order,dyn = False,L=False):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    spread1 = standard(spread[:(166-D)])
    model = AutoReg(spread1,lags = order)
    fit = model.fit()
    print(fit.params)
    Arima = ARIMA(spread1,(order,0,0))
    fit2 = Arima.fit(method='css')
    print('Arima bic:',fit2.bic)
    print('bic:',fit.bic)
    if L:
        pre = fit.predict(start = order,end = L,dynamic = dyn)
    else:
        pre = fit.predict(start = order,end = (len(spread1)-1),dynamic = dyn)
    plt.figure(figsize=(10,4))
    plt.plot(spread1[order:],color='b',label='origin')
    if dyn:
        plt.plot(pre,color='r',label='ARfit')
    else:
        plt.plot(pre,color='r',label='Forecast1')
    plt.axhline(0,color='black',label='mean')
    plt.axhline(1,color='orange',label='Open thres')
    plt.axhline(-1,color='orange')
    plt.legend()
    if L:
        pass
    else:
        n = len(spread1)
        mse = mean_squared_error(spread1[order:],pre)
        print('MSE:',mse)
        print('BIC count:',order*np.log(n)/n + np.log(n*mse/(n-order)))

def check_spread_converge(output,num,model,lastNum,D=16,std=False,times=10):
    spread_m,_,p = spread_mean(output,num,model,D,order=True)
    mean = np.mean(spread_m[-lastNum:])
    if std == False:
        std = output[2*num,6]
    distance = np.abs(spread_m-mean)
    converge = np.zeros(times)
    for i in range(times):
        converge[i] = p + np.argmax(distance < std/(i+1))
        if converge[i] == 0:
            if distance[0]>std/(i+1):
                converge[i] = 150
    return converge

def check_spread_Absconverge(output,num,model,lastNum,D=16,std=False,times=10):
    spread_m,_,p = spread_mean(output,num,model,D,order=True)
    mean = np.mean(spread_m[-lastNum:])
    if std == False:
        std = output[2*num,6]
    distance = np.abs(spread_m-mean)
    converge = np.zeros(times)
    for i in range(times):
        inside = (distance-std/(i+1)) > 0
        converge[i] = 150-np.argmax(inside[::-1])
        if converge[i] == 0:
            if distance[0]>std/(i+1):
                converge[i] = 150
    return converge

def check_mean_bias(output,num,model,lastNum,D=16):
    spread_m,_ = spread_mean(output,num,model,D)
    mean = np.mean(spread_m[-lastNum:])
    std = output[2*num,6]
    o_mean = output[2*num,5]
    distance = np.abs(o_mean-mean)/std
    return distance

def sharpe_test(output,i,SprReward_03,D=16):
    spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
    std = np.std(spread)
    sharpe = SprReward_03[i]/std
    return sharpe

我擋
####start
i=1
D=16
m1,_=ex_spread_mean(output,i,model,D,248,ini=10)
m4,_=ex_spread_mean(output,i,model,D,248,ini=40)
m10,_=ex_spread_mean(output,i,model,D,248,ini=100)
m18,_=ex_spread_mean(output,i,model,D,248,ini=180)
spread = output[2*i,3]*np.log(output[2*i,8:(274-D)]) + output[(2*i+1),3]*np.log(output[(2*i+1),8:(274-D)])
spread = spread[2:]
mu = output[2*i,5]
std = output[2*i,6]
plt.figure(figsize=(16,4))
plt.axhline(mu)
plt.axhline(mu+1.5*std,color='r')
plt.axhline(mu-1.5*std,color='r')
plt.title('num:{}'.format(i))
plt.plot(np.arange(10,246),m1,label='10')
plt.plot(np.arange(40,246),m4,label='40')
plt.plot(np.arange(100,246),m10,label='100')
plt.plot(np.arange(180,246),m18,label='180')
plt.plot(spread,color = 'b',alpha=0.4)
plt.legend()

two_select = np.intersect1d(select_save[0],Cselect_save[4])
sum(record[two_select]==666)/len(two_select)
sum(Reward_03[two_select])
sum(SprReward_03[two_select])
len(two_select)

count = 0
error = 0
unit = 0
for num in range(len(Reward_03)):
    eva,_,eva2 = get_IBA(output,num,table.model_type[num],Ori=True)
    if abs(eva[0]-eva2[0])<1e10 or abs(eva[0]-eva2[1])<1e10:
        count+=1
    else:
        error+=1
        print(num)
    if (eva2[0]-1)<1e10 or (eva2[1]-1)<1e10:
        unit+=1

num = 116
table.model_type[num]
num = select[0][num]
print(record[num])
print(table.model_type[num])
i=0
for num in Cselect_save[0]:
    i = i+1
    if i>=0:
        spread_mean_plot(output,num,table.model_type[num],16,ini=0,future=True)
#        m,_=spread_mean(output,num,table.model_type[num],16)
#        plt.axhline(m[-1],color='g',ls='--',label='new_close')
#        plt.axhline(m[-1]+1.5*table.stdev.iloc[num],color='purple',ls='--',label='new_open')
#        plt.axhline(m[-1]-1.5*table.stdev.iloc[num],color='purple',ls='--')
#        plt.legend()        
        if i>=20:
            break

absDif = 0
count = 0
for num in np.setdiff1d(select_save[-1],select_save[30]):
    print('num',num)
    Estd = get_Estd(output,num,table.model_type[num])
    print('Estd:',Estd)
    print('std:',table.stdev.iloc[num])
    absDif += np.abs(Estd-table.stdev.iloc[num])
    count += 1
print('avg difference:',absDif/count)
print('times:',count)

absDif = 0
count = 0
for num in select_save[0]:
    print('num',num)
    Estd = get_Estd(output,num,table.model_type[num])
    print('Estd:',Estd)
    print('std:',table.stdev.iloc[num])
    absDif += np.abs(Estd-table.stdev.iloc[num])
    count += 1
print('avg difference:',absDif/count)
print('times:',count)

only_mean_plot(output,num,table.model_type[num],500,16,True)
for num in select[0]:
    temp,_ = get_root(output,num,table.model_type[num],16)
    print('now:',num,',cond:',temp)

temp,_=spread_mean(output,num,table.model_type[num],D=16)
temp[-10:]

ARSave(output,num,16,1)
MarkovSwitching(output,num,16)
BS_check(output,num,table)
ARplot(output,num,16,1,True)
fftTraf(output,num,16)
spread_coef(output,num,table.model_type[num],16)
check_spread_converge(output,num,table.model_type[num],16,10)
check_spread_Absconverge(output,num,table.model_type[num],16,10)
#save_to_csv(output,2,16,True)
#save_to_csv(output,3,16,False)

condiction = np.zeros(len(output))
for i in range(len(output)//2):
    if i%10 == 0:
        print(i)
    num = MarkovCheck(output,i,16)
    condiction[2*i] = num
    condiction[2*i+1] = num
Mar = output[condiction==1,:]
print('Mar len:',len(Mar)//2)
print('win times:',sum(Mar[:,0]==666)//2)
print(sum([Mar[2*i,7] for i in range(len(Mar)//2)]))
print(sum(Reward_03)*(len(Mar)//2)/len(Reward_03))

#np.mean(Reward_03[Reward_03>0])
#np.mean(Reward_03[Reward_03<0])

SprReward_03 = SprReward-0.003
BicCompare = np.zeros([len(output)//2,4])
for i in range(len(output)//2):
    if i%10 == 0:
        print(i)
    #BicCompare[i,1] = MarkovSave(output,i,16,1)
    #BicCompare[i,1] = ARSave(output,i,16,1)
    BicCompare[i,1] = MyBicSave(output,i,16,1,False)
    #BicCompare[i,1] = AutoRegSave(output,i,16,1)
    #BicCompare[i,1] = ARIMASave(output,i,16,1)
    BicCompare[i,0] = output[2*i,0]
    BicCompare[i,2] = output[2*i,7]    
    BicCompare[i,3] = SprReward_03[i]

threshold,Positive,Negative = threshold_plot(BicCompare,1000)
#plt.plot(threshold,(Positive-Negative),label = 'Dif of PRr and NRr')
plt.plot(threshold,(Positive-Negative),label = 'Dif of PLN and NLN')
plt.legend()
thr = -2.1
print('len:',sum(BicCompare[:,1]>thr))
print('strategy win:',sum(BicCompare[BicCompare[:,1]>thr,0]==666))
print('Profit:',sum(BicCompare[BicCompare[:,1]>thr,2]))
print('Profit win:',sum(BicCompare[BicCompare[:,1]>thr,2]>0))
plt.hist(BicCompare[BicCompare[:,1]>thr,2])

sort_order = np.argsort(BicCompare[:,1])
sort_BicCompare = BicCompare[sort_order,:]
plt.plot(sort_BicCompare[:,1],sort_BicCompare[:,2])

num = 5
step = (sort_BicCompare[-1,1]-sort_BicCompare[0,1])/num
temp = sort_BicCompare.copy()
now_step = sort_BicCompare[0,1]
profit_mean = np.zeros(num)
return_mean = np.zeros(num)
MR_count = np.zeros(num)
x = np.zeros(num)
for i in range(num):
    x[i] = now_step+step/2
    now_step += step
    use = temp[temp[:,1]<=now_step]
    temp = temp[temp[:,1]>now_step]
    print(len(use))
    if np.mean(use[:,2]) != np.nan:
        profit_mean[i] = np.mean(use[:,2])
        return_mean[i] = np.mean(use[:,3])
        MR_count[i] = np.mean(use[:,0]==666)
    else:
        print('error on:',i)

plt.plot(x,profit_mean)
for i in range(num-1):
    plt.axvline(x[i]+step/2,linestyle = '--',color = 'black')
plt.axhline(0,linestyle = '--',color='r')

plt.plot(x,return_mean)
for i in range(num-1):
    plt.axvline(x[i]+step/2,linestyle = '--',color = 'black')
plt.axhline(0,linestyle = '--',color='r')

plt.plot(x,MR_count)
for i in range(num-1):
    plt.axvline(x[i]+step/2,linestyle = '--',color = 'black')
plt.axhline(0.5,linestyle = '--',color='r')

def mtoCsv(meanB_Mprof,meanB_prof,name):
    M = np.concatenate([meanB_Mprof,meanB_prof],axis=1)
    M = pd.DataFrame(M[:,:-1])
    c_name = ['Profit(0.3%) mean','return mean','Normal Close Rate','times','Profit(0.3%)']
    M.columns = c_name
    M.to_csv('MeanDifThres{}{}.csv'.format(name,y),index=False)

def CtoCsv(converge_Mprof,converge_prof,name):
    C = np.concatenate([converge_Mprof,converge_prof],axis=1)
    C = pd.DataFrame(C[:,:-1])
    c_name = ['Profit(0.3%) mean','return mean','Normal Close Rate','times','Profit(0.3%)']
    C.columns = c_name
    C.to_csv('ConvergeThres{}{}.csv'.format(name,y),index=False)


#root select
SprReward_03 = SprReward-0.003
root_select = np.zeros([len(SprReward_03)])
root = []
for i in range(len(SprReward_03)):
    r,e = get_root(output,i,table.model_type[i],16)
    root.append(r)
    root_select[i] = e
select = np.where(root_select==0)
print('len:',len(select[0]))
print('Reward_03:',sum(Reward_03[select]))
print('Normal Close Rate:',sum(record[select]==666)/len(select[0]))
distance = np.zeros(len(select[0]))
for i in range(len(distance)):
    distance[i] = check_mean_bias(output,select[0][i],table.model_type[i],10)
meanB_Mprof = np.zeros([50,4])
meanB_prof = np.zeros([50,2])
select_save = []
for i in range(50):
    mselect = np.where( distance < 0.02*(i+1) )
    meanB_Mprof[i,0] = np.mean(Reward_03[select[0][mselect]])
    meanB_Mprof[i,1] = np.mean(SprReward_03[select[0][mselect]])
    meanB_Mprof[i,2] = np.mean(record[select[0][mselect]]==666)
    meanB_Mprof[i,3] = len(mselect[0])
    select_save.append(select[0][mselect])
    meanB_prof[i,0] = sum(Reward_03[select[0][mselect]])
    meanB_prof[i,1] = sum(SprReward_03[select[0][mselect]])

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(0.02,1+0.02,0.02),meanB_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(0.02,1+0.02,0.02),meanB_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(0.02,1+0.02,0.02),meanB_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)

#SprReward_03 = SprReward-0.003
#sharpe = np.zeros(len(SprReward_03))
#for i in range(len(sharpe)):
#    sharpe[i] = sharpe_test(output,i,SprReward_03)

SprReward_03 = SprReward-0.003
times = 10
converge = np.zeros([len(select[0]),times])
for i in range(len(converge)):
    #converge[i,:] = check_spread_converge(output,select[0][i],table.model_type[i],10,std=0.0005,times=times)
    #converge[i,:] = check_spread_converge(output,select[0][i],table.model_type[i],10)
    converge[i,:] = check_spread_Absconverge(output,select[0][i],table.model_type[i],10)
converge_Mprof = np.zeros([46,4])
converge_prof = np.zeros([46,2])
Cselect_save = []
for i in range(46):
    cselect = np.where( converge[:,5] < 2*(i+5) )
    #cselect = np.where( converge[:,10] < 2*(i+1) )
    converge_Mprof[i,0] = np.mean(Reward_03[select[0][cselect]])
    converge_Mprof[i,1] = np.mean(SprReward_03[select[0][cselect]])
    converge_Mprof[i,2] = np.mean(record[select[0][cselect]]==666)
    converge_Mprof[i,3] = len(cselect[0])
    converge_prof[i,0] = sum(Reward_03[select[0][cselect]])
    converge_prof[i,1] = sum(SprReward_03[select[0][cselect]])
    Cselect_save.append(select[0][cselect])

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(10,102,2),converge_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(10,102,2),converge_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(10,102,2),converge_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)

mtoCsv(meanB_Mprof,meanB_prof,'WithRootSelect')
CtoCsv(converge_Mprof,converge_prof,'WithRootSelect')

RsmMp2018 = meanB_Mprof.copy()
Rsmp2018 = meanB_prof.copy()
RscMp2018 = converge_Mprof.copy()
Rscp2018 = converge_prof.copy()

#mean bias
SprReward_03 = SprReward-0.003
distance = np.zeros(len(table))
for i in range(len(table)):
    distance[i] = check_mean_bias(output,i,table.model_type[i],1)
meanB_Mprof = np.zeros([50,4])
meanB_prof = np.zeros([50,2])
select_save = []
for i in range(50):
    select = np.where( distance < 0.02*(i+1) )
    meanB_Mprof[i,0] = np.mean(Reward_03[select])
    meanB_Mprof[i,1] = np.mean(SprReward_03[select])
    meanB_Mprof[i,2] = np.mean(record[select]==666)
    meanB_Mprof[i,3] = len(select[0])
    select_save.append(select[0])
    meanB_prof[i,0] = sum(Reward_03[select])
    meanB_prof[i,1] = sum(SprReward_03[select])

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(0.02,1+0.02,0.02),meanB_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(0.02,1+0.02,0.02),meanB_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(0.02,1+0.02,0.02),meanB_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)

##Sharpe ratio
#SprReward_03 = SprReward-0.003
#sharpe = np.zeros(len(SprReward_03))
#for i in range(len(sharpe)):
#    sharpe[i] = sharpe_test(output,i,SprReward_03)
    
####converge
SprReward_03 = SprReward-0.003
times=10
converge = np.zeros([len(table),times])
for i in range(len(table)):
    #converge[i,:] = check_spread_converge(output,i,table.model_type[i],10,std=0.0005,times=times)
    #converge[i,:] = check_spread_converge(output,i,table.model_type[i],1)
    converge[i,:] = check_spread_Absconverge(output,i,table.model_type[i],1)
converge_Mprof = np.zeros([46,5])
converge_prof = np.zeros([46,2])
Cselect_save = []
for i in range(46):
    select = np.where( converge[:,5] < 2*(i+5) )
    #select = np.where( converge[:,10] < 2*(i+1) )
    converge_Mprof[i,0] = np.mean(Reward_03[select])
    converge_Mprof[i,1] = np.mean(SprReward_03[select])
    converge_Mprof[i,2] = np.mean(record[select]==666)
#    converge_Mprof[i,3] = np.mean(sharpe[select])
    converge_Mprof[i,3] = np.mean(Reward_03[select]>0)
    converge_Mprof[i,4] = len(select[0])
    converge_prof[i,0] = sum(Reward_03[select])
    converge_prof[i,1] = sum(SprReward_03[select])
    Cselect_save.append(select[0])

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(10,102,2),converge_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(10,102,2),converge_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(10,102,2),converge_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)
#axs[3].plot(np.arange(5,105,5),converge_prof[:,3])
#axs[3].set_title('sharpe mean',fontweight="bold", size=20)

#mMp2018 = meanB_Mprof.copy()
#mp2018 = meanB_prof.copy()
#cMp2018 = converge_Mprof.copy()
#cp2018 = converge_prof.copy()
#
#mtoCsv(meanB_Mprof,meanB_prof,'OnlySymmestry')
#CtoCsv(converge_Mprof,converge_prof,'EEOnlySymmestry')

#前十名
#np.argsort(Rscp2015[:,0])[-10:]
#np.argsort(Rscp2016[:,0])[-10:]
#np.argsort(Rscp2017[:,0])[-10:]
#np.argsort(Rscp2018[:,0])[-10:]


#分割看
intv = 10
long = 0.5
converge_Mprof = np.zeros([intv,4])
converge_prof = np.zeros([intv,2])
Cselect_save = []
for i in range(intv):
    select = np.where( distance < (long/intv)*(i+1) )
    select1 = np.where( (long/intv)*i <= distance )
    select = np.intersect1d(select,select1)
    converge_Mprof[i,0] = np.mean(Reward_03[select])
    converge_Mprof[i,1] = np.mean(SprReward_03[select])
    converge_Mprof[i,2] = np.mean(record[select]==666)
#    converge_Mprof[i,3] = np.mean(sharpe[select])
    converge_Mprof[i,3] = len(select)
    converge_prof[i,0] = sum(Reward_03[select])
    converge_prof[i,1] = sum(SprReward_03[select])
    Cselect_save.append(select)

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(long/intv,long+long/intv,(long/intv)),converge_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(long/intv,long+long/intv,(long/intv)),converge_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(long/intv,long+long/intv,(long/intv)),converge_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)


intv = 5
long = 70
converge_Mprof = np.zeros([intv,4])
converge_prof = np.zeros([intv,2])
Cselect_save = []
for i in range(intv):
    select = np.where( converge[:,5] < (long/intv)*(i+1) )
    select1 = np.where( (long/intv)*i <= converge[:,5] )
    select = np.intersect1d(select,select1)
    converge_Mprof[i,0] = np.mean(Reward_03[select])
    converge_Mprof[i,1] = np.mean(SprReward_03[select])
    converge_Mprof[i,2] = np.mean(record[select]==666)
#    converge_Mprof[i,3] = np.mean(sharpe[select])
    converge_Mprof[i,3] = len(select)
    converge_prof[i,0] = sum(Reward_03[select])
    converge_prof[i,1] = sum(SprReward_03[select])
    Cselect_save.append(select)

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(long/intv,long+long/intv,(long/intv)),converge_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(long/intv,long+long/intv,(long/intv)),converge_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(long/intv,long+long/intv,(long/intv)),converge_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)

########
#IBA converge
IBA_save = np.zeros(len(table))
for i in range(len(table)):
    temp,_ = get_IBA(output,i,table.model_type[i])
    IBA_save[i] = temp[0]

itv = 1000
IBA_Mprof = np.zeros([itv,4])
IBA_prof = np.zeros([itv,2])
IBA_select = []
for i in range(itv):
    select = np.where( IBA_save < (0 + 1*(i+1)/itv) )
    IBA_Mprof[i,0] = np.mean(Reward_03[select])
    IBA_Mprof[i,1] = np.mean(SprReward_03[select])
    IBA_Mprof[i,2] = np.mean(record[select]==666)
#    converge_Mprof[i,3] = np.mean(sharpe[select])
    IBA_Mprof[i,3] = len(select[0])
    IBA_prof[i,0] = sum(Reward_03[select])
    IBA_prof[i,1] = sum(SprReward_03[select])
    IBA_select.append(select[0])

fig, axs = plt.subplots(3, 1)
fig.set_figheight(12)
fig.set_figwidth(8)
axs[0].plot(np.arange(1,itv+1),IBA_Mprof[:,0])
axs[0].set_title('Profit(0.3%) mean',fontweight="bold", size=20)
axs[1].plot(np.arange(1,itv+1),IBA_Mprof[:,1])
axs[1].set_title('return mean',fontweight="bold", size=20)
axs[2].plot(np.arange(1,itv+1),IBA_Mprof[:,2])
axs[2].set_title('Normal Close Rate',fontweight="bold", size=20)

#####year research####
cbdR = pd.DataFrame(table.date)
cbdR['Reward_03'] = Reward_03
cbdR['date'] = pd.to_datetime(cbdR['date'])
#cbdR['record'] = record==666
group = cbdR.groupby('date')
groupsum = group.sum()
groupcumsum = groupsum.cumsum()
groupmean = group.mean()
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%Y%m')
fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(15)
ax.plot(groupcumsum.index,groupcumsum['Reward_03'])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

date = np.argsort(table.Estd).iloc[-24:]
print(set(table.date.iloc[date]))
plt.hist(table.Estd)

tw = pd.read_csv('TWII.csv',index_col=0)
tw.index = pd.to_datetime(tw.index)
twR = (tw['High']-tw['Low'])/tw['Low']
years = ['2015','2016','2017','2018']
for i in range(len(years)):
    print('{} mean'.format(years[i]),np.mean(twR[years[i]]))
    print('{} median'.format(years[i]),np.nanmedian(twR[years[i]]))

year = '2018'
plt.figure(figsize=(15,10))
plt.plot(tw[year]['Adj Close'])
for line in np.where(groupsum[year]['Reward_03']>10)[0]:
    plt.axvline(groupsum[year].index[line],
                color='r',linestyle='--')

plt.figure(figsize=(15,10))
plt.plot(tw['2015':'2018']['Adj Close'])
for line in np.where(groupsum['2015':'2018']['Reward_03']>10)[0]:
#    plt.axvline(groupsum['2015':'2018'].index[line],
#                color='r',linestyle='--')
    day = groupsum['2015':'2018'].index[line]
    plt.plot(day,tw.loc[day]['Adj Close'], marker='*',markersize=10,color="red")
    print('date:{},profit:{}'.format(groupsum['2015':'2018'].index[line],
          groupsum['2015':'2018']['Reward_03'].iloc[line]))

plt.figure(figsize=(15,10))
plt.plot(tw['2015':'2018']['Adj Close'])
for line in np.where(groupsum['2015':'2018']['Reward_03']<-20)[0]:
#    plt.axvline(groupsum['2015':'2018'].index[line],
#                color='b',linestyle='--')
    day = groupsum['2015':'2018'].index[line]
    plt.plot(day,tw.loc[day]['Adj Close'], marker='*',markersize=10,color="green")
    print('date:{},profit:{}'.format(groupsum['2015':'2018'].index[line],
          groupsum['2015':'2018']['Reward_03'].iloc[line]))

plt.figure(figsize=(15,10))
plt.plot(tw['2015':'2018']['Adj Close'])
for d in np.argsort(table.Estd)[-int(len(table)/5):]:
    day = datetime.strptime(table.date.iloc[d],'%Y-%m-%d')
    plt.plot(day, tw.loc[day]['Adj Close'], marker='o', markersize=5, color="red")
plt.axvline('2018-10-09',color='b',linestyle='--')
plt.axvline('2018-07-09',color='b',linestyle='--')
plt.axvline('2015-08-24',color='b',linestyle='--')

##############################
temp = np.zeros(20)
plt.figure(figsize=(16,4))
for coef in np.arange(-0.8,0.9,0.3):
    temp[0] = 20+np.random.rand(1)
    for i in range(1,20):
        temp[i] = 3+coef*temp[i-1]+np.random.rand(1)
    plt.plot(temp,label = 'coef:{}'.format(coef))
plt.legend(loc='upper right')

temp = np.zeros(150)
temp[0] = np.random.rand(1)
temp[1] = np.random.rand(1)+0.3*temp[0]
for i in range(2,150):
    temp[i] = 0.05*i+0.3*temp[i-1]+0.3*temp[i-2]+np.random.rand(1)
plt.plot(temp)
temp = np.zeros(150)
temp[0] = np.random.rand(1)
for i in range(1,150):
    temp[i] = 0.3*temp[i-1]+np.random.rand(1)

temp = standard(temp)
#t1 = ARIMA(temp,(1,0,0))
t1 = AutoReg(temp,1)
tt1 = t1.fit()
pre = tt1.predict(start = 1,end = (len(temp)-1),dynamic = False)
plt.figure(figsize=(10,4))
plt.plot(temp[1:],color='b',label='origin')
plt.plot(pre,color='r',label='ARfit')
#plt.axhline(np.mean(temp),color='black')
plt.axhline(np.mean(temp)+np.std(temp),color='orange')
plt.axhline(np.mean(temp)-np.std(temp),color='orange')
plt.legend()
print('1:',tt1.aic)
t2 = ARIMA(temp,(2,0,0))
tt2 = t2.fit()
pre = tt2.predict(start = 2,end = (len(temp)-1),dynamic = True)
plt.figure(figsize=(10,4))
plt.plot(temp[2:],color='b',label='origin')
plt.plot(pre,color='r',label='ARfit')
#plt.axhline(np.mean(temp),color='black')
plt.axhline(np.mean(temp)+np.std(temp),color='orange')
plt.axhline(np.mean(temp)-np.std(temp),color='orange')
plt.legend()
print('2:',tt2.aic)
t3 = ARIMA(temp,(3,0,0))
tt3 = t3.fit()
pre = tt3.predict(start = 3,end = (len(temp)-1),dynamic = True)
plt.figure(figsize=(10,4))
plt.plot(temp[3:],color='b',label='origin')
plt.plot(pre,color='r',label='ARfit')
#plt.axhline(np.mean(temp),color='black')
plt.axhline(np.mean(temp)+np.std(temp),color='orange')
plt.axhline(np.mean(temp)-np.std(temp),color='orange')
plt.legend()
print('3:',tt3.aic)
t4 = ARIMA(temp,(4,0,0))
tt4 = t4.fit()
pre = tt4.predict(start = 4,end = (len(temp)-1),dynamic = True)
plt.figure(figsize=(10,4))
plt.plot(temp[4:],color='b',label='origin')
plt.plot(pre,color='r',label='ARfit')
#plt.axhline(np.mean(temp),color='black')
plt.axhline(np.mean(temp)+np.std(temp),color='orange')
plt.axhline(np.mean(temp)-np.std(temp),color='orange')
plt.legend()
print('4:',tt4.aic)

temp = np.zeros(150)
temp[0] = 5 + np.random.rand(1)
lc = 5
for i in range(1,50):
    temp[i] = 5+0.3*(temp[i-1]-lc)+np.random.rand(1)
    lc = 5
for i in range(50,100):
    temp[i] = 1+0.3*(temp[i-1]-lc)+np.random.rand(1)
    lc = 1
for i in range(100,150):
    temp[i] = 3+0.3*(temp[i-1]-lc)+np.random.rand(1)
    lc = 3
plt.plot(temp)
temp = standard(temp)

#count table
st_name = set()
for i in range(len(table)):
    st_name.add(str(table.iloc[i,0]))
    st_name.add(str(table.iloc[i,1]))
st_name = list(st_name)
table['R'] = Reward_03
count_table = pd.DataFrame([])
count_table['times'] = np.zeros(len(st_name))
count_table.index = st_name
count_dic = defaultdict(list)

for i in range(len(table)):
    if table.iloc[i,12] >0:
        count_table.loc[str(table.iloc[i,0])] +=1
        count_table.loc[str(table.iloc[i,1])] +=1
        count_dic[str(table.iloc[i,0])].append(i)
        count_dic[str(table.iloc[i,1])].append(i)
        
count_table = count_table.sort_values(by='times',ascending=False)
num = 147
BS_check(output,num,table)
print(Reward_03[num])

#Reward<->retrun distribution
SprReward_03 = SprReward-0.003
table['return'] = SprReward_03
sort_table = table.sort_values(by='return',ascending=False)
sort_table = sort_table.reset_index()
sort_output = np.zeros(output.shape)
sort_order = np.argsort(SprReward_03)[::-1]
for i in range(len(sort_order)):
    sort_output[2*i,:] = output[2*sort_order[i],:]
    sort_output[(2*i+1),:] = output[(2*sort_order[i]+1),:]

start = 20
BS_5check(sort_output,start,sort_table,SprReward_03,sort_order)

t2 = sm.tsa.MarkovAutoregression(temp, k_regimes=2, order=1,
                                 switching_ar=False,
                                 switching_trend=True,
                                 switching_exog=False,
                                 switching_variance=False)
fit = t2.fit()
print('markov2:',fit.bic)
t3 = sm.tsa.MarkovAutoregression(temp, k_regimes=3, order=1,
                                 switching_ar=False,
                                 switching_trend=True,
                                 switching_exog=False,
                                 switching_variance=False)

fit = t3.fit()
print('markov3:',fit.bic)
t4 = sm.tsa.MarkovAutoregression(temp, k_regimes=4, order=1,
                                 switching_ar=False,
                                 switching_trend=True,
                                 switching_exog=False,
                                 switching_variance=False)
fit = t4.fit()
print('markov4:',fit.bic)

positive = output[output[:,7]>0]
success = output[output[:,0]==666]
false = output[output[:,0]!=666]

fftTraf(negative,50)
fftTraf(positive,10)

fftTraf(false,14)
fftTraf(success,2)

Sy1,Sy2,Sdif,SSdif = fftCheck(success,4)
Fy1,Fy2,Fdif,FSdif = fftCheck(false,4)
Sy1,Sy2,Sdif = fftHalfCheck(success,4)
Fy1,Fy2,Fdif = fftHalfCheck(false,4)

plt.hist(SSdif,bins = np.arange(-1,3,0.25))
plt.hist(FSdif,bins = np.arange(-1,3,0.25))

Stemp = np.arange(len(SSdif))[np.abs(Sdif)>0.75]
Ssum = []
for i in range(len(Stemp)):
    Ssum.append(success[2*Stemp[i],7])
sum(Ssum)

Ftemp = np.arange(len(FSdif))[np.abs(Fdif)>0.75]
Fsum = []
for i in range(len(Ftemp)):
    Fsum.append(false[2*Ftemp[i],7])
sum(Fsum)

fftTraf(success,Stemp[5])
fftTraf(false,Ftemp[25])


block_area = areaCheck(save_array)
plt.hist(block_area)
area = areaCheck(output)
plt.hist(area)

block_area = stdCheck(save_array)
plt.hist(block_area)
area = stdCheck(output)
plt.hist(area)

block_area = zero(save_array)
plt.hist(block_area)
area = zero(output)
plt.hist(area)

plotSprV1V2(6,save_array)
plotSprV1V2(5,output)

plotSprV1V2(2,check)
plotSprV1V2(1,small_check)

np.median(np.abs(small_check[:,5]))
np.median(np.abs(check[:,5]))
np.mean(np.abs(small_check[:,5]))
np.mean(np.abs(check[:,5]))

np.median(np.abs(small_check[:,6]))
np.median(np.abs(check[:,6]))
np.mean(np.abs(small_check[:,6]))
np.mean(np.abs(check[:,6]))

positive = output[output[:,7]>0]
negative = output[output[:,7]<0]

pos_overzero = zero(positive)
plt.hist(pos_overzero)
neg_overzero = zero(negative)
plt.hist(neg_overzero)

plotSprV1V2(4,positive)

np.mean(np.abs(positive[:,5]))
np.mean(np.abs(negative[:,5]))
np.median(np.abs(positive[:,5]))
np.median(np.abs(negative[:,5]))

np.mean(np.abs(positive[:,6]))
np.mean(np.abs(negative[:,6]))
np.median(np.abs(positive[:,6]))
np.median(np.abs(negative[:,6]))

np.mean(np.abs(positive[:,1]))
np.mean(np.abs(negative[:,1]))
np.median(np.abs(positive[:,1]))
np.median(np.abs(negative[:,1]))

