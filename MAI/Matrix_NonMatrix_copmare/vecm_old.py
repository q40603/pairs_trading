#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 20:31:47 2019

@author: chaohsien
"""

import pandas as pd
import numpy as np
from scipy.linalg import eigh 

#data = da1
#data = pd.read_csv('20180905_averagePrice_min.csv',encoding='utf-8')
#da1 = day1_1.iloc[:,1:3]

# 估計VECM參數，算出特徵向量與特徵值

#model = 'H2'
#p = 3              # VAR落後期數

def vecm(data,model,p):
    
    k = len(data.T)     # k檔股票

    dY_all = data.diff().drop([0]) ; dY_all.index  = np.arange(0,len(dY_all),1)
    dY = dY_all.iloc[p-1:len(data),:]
    
    Y_1 = data.iloc[p-1:len(data)-1,:] ; Y_1.index  = np.arange(0,len(Y_1),1)
    
    if model == 'H1*':
    
        Y_1 = np.hstack((np.array(Y_1),np.ones([len(dY),1])))
        
    if model == 'H*':
        
        time = np.arange(1,len(dY)+1,1).reshape((len(dY),1))
        
        Y_1 = np.hstack((np.array(Y_1), time ))

    dX = np.zeros([len(dY),k*(p-1)])
    for i in range(p-1):
    
        dX[:,i*k:k*(i+1)] = np.array(dY_all.iloc[(p-i-2):len(data)-i-2,:])
    
    if model == 'H1' or model == 'H*':

        dX = np.hstack((dX,np.ones([len(dY),1])))
    
    M = np.eye(len(dY)) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )

    R0 = np.dot( dY.T , M )
    R1 = np.dot( Y_1.T , M )

    S00 = np.dot( R0 , R0.T ) / len(dY)
    S01 = np.dot( R0 , R1.T ) / len(dY)
    S10 = np.dot( R1 , R0.T ) / len(dY)
    S11 = np.dot( R1 , R1.T ) / len(dY)

    A = np.dot(S10 , np.dot(np.linalg.inv(S00) , S01))
    
    eigvals, eigvecs = eigh(A , S11 , eigvals_only=False)
 
    testStat = -2 * np.log(np.cumprod(1-eigvals) ** (len(dY)/2))
    
    return [ testStat , eigvals , eigvecs ]

def rank(data,model,p):
    
    testStat = vecm( data ,  model , p )[0]       # 第零個位置是檢定統計量
    k = len(data.T)                               # k檔股票
    
    if model == 'H2':

        if k == 2:
    
            if testStat[1] < 12.3329:
        
                rank = 0
        
            elif testStat[0] < 4.1475:
        
                rank = 1
        
            else:
        
                rank = 2

    elif model == 'H1*':
    
        if k == 2:
    
            if testStat[2] < 20.3032:
        
                rank = 0
        
            elif testStat[1] < 9.1465:
        
                rank = 1
        
            else:
        
                rank = 2

    elif model == 'H1':
    
        if k == 2:
    
            if testStat[1] < 15.4904:
        
                rank = 0
        
            elif testStat[0] < 3.8509:
        
                rank = 1
        
            else:
        
                rank = 2
                
    else: # model == 'H*'
        
        if k == 2:
    
            if testStat[2] > 25.8863:
        
                rank = 0
        
            elif testStat[1] < 12.5142:
        
                rank = 1
        
            else:
        
                rank = 2

    return rank

def eig(data,model,p,rank):

    eigvals = vecm( data , model , p )[1]   # 第一個位置是特徵值
    k = len(data.T)                         # k檔股票
    
    if model == 'H2':

        if k == 2:
    
            eigva = eigvals[0]
                
    elif model == 'H1*':
    
        if k == 2:
            
            eigva = eigvals[1]
            
    elif model == 'H1':
    
        if k == 2:
            
            eigva = eigvals[0]
            
    else: # model == 'H*'
        
        if k == 2:
            
            eigva = eigvals[1]

    return eigva

def weigh(data,model,p,rank):
    
    eigvecs = vecm( data , model , p )[2]     # 第2個位置是特徵向量
    k = len(data.T)                           # k檔股票
    
    if model == 'H2':

        if k == 2:
    
            wei = -eigvecs[:,1]
                
    elif model == 'H1*':
    
        if k == 2:
            
            wei = eigvecs[:,2] 
            
    elif model == 'H1':
    
        if k == 2:
            
           wei = eigvecs[:,1] 
           
    else: # model == 'H*'
        
        if k == 2:
            
            wei = eigvecs[:,2] 

    return wei.T

def para_vecm(data,model,p):
    
    data = pd.DataFrame(data)
    
    k = len(data.T)     # k檔股票

    dY_all = data.diff().drop([0]) ; dY_all.index  = np.arange(0,len(dY_all),1)
    dY = dY_all.iloc[p-1:len(data),:]
    
    Y_1 = data.iloc[p-1:len(data)-1,:] ; Y_1.index  = np.arange(0,len(Y_1),1)
    
    if model == 'H1*':
    
        Y_1 = np.hstack((np.array(Y_1),np.ones([len(dY),1])))
        
    if model == 'H*':
        
        time = np.arange(1,len(dY)+1,1).reshape((len(dY),1))
        
        Y_1 = np.hstack((np.array(Y_1), time ))

    dX = np.zeros([len(dY),k*(p-1)])
    for i in range(p-1):
    
        dX[:,i*k:k*(i+1)] = np.array(dY_all.iloc[(p-i-2):len(data)-i-2,:])
    
    if model == 'H1' or model == 'H*':

        dX = np.hstack((dX,np.ones([len(dY),1])))
    
    M = np.eye(len(dY)) - np.dot(np.dot(dX , np.linalg.inv(np.dot(dX.T , dX))) , dX.T )

    R0 = np.dot( dY.T , M )
    R1 = np.dot( Y_1.T , M )

    S00 = np.dot( R0 , R0.T ) / len(dY)
    S01 = np.dot( R0 , R1.T ) / len(dY)
    S10 = np.dot( R1 , R0.T ) / len(dY)
    S11 = np.dot( R1 , R1.T ) / len(dY)

    A = np.dot(S10 , np.dot(np.linalg.inv(S00) , S01))
    
    eigvals, eigvecs = eigh(A , S11 , eigvals_only=False)
    
    if model == 'H2':
        
        beta = np.mat(-eigvecs[:,1]).T
        
    elif model == 'H1':
        
        beta = np.mat(eigvecs[:,1]).T
        
    else :
        
        beta = np.mat(eigvecs[:,2]).T                                  # 共整合係數
    
    alpha = np.dot(S01,beta)
    
    D = np.dot( np.dot(dY.T - np.dot(np.dot(alpha,beta.T),Y_1.T) , dX ) , np.linalg.inv(np.dot(dX.T,dX)) )
    
    at = dY.T - np.dot(np.dot(alpha,beta.T),Y_1.T) + np.dot(D,dX.T)    # 殘差
    
    #sigma_u = np.dot(at,at.T)/len(data)                                # 殘差共變異數
    
    if model == 'H1':
        
        intercept = np.mat(D[:,len(D.T)-1]).T
        D = D[:,:-1]
        
    elif model == 'H1*':
        
        intercept = alpha * beta[len(beta)-1,:]
        beta = beta[:-1]
    
    else:
        
        intercept = np.zeros((2,1))
    
    # VAR representation 
    if p == 1:
        
        A1 = np.eye(k) + np.dot(alpha,beta.T)
        
    elif p == 2:
        
        A1 = np.eye(k) + np.dot(alpha,beta.T) + D
        A1 = np.hstack((A1,D))
        
    else:
        
        A1 = np.eye(k) + np.dot(alpha,beta.T) + D[:,0:2]
        for i in range(p-2):
            
            A2 = D[:,2*(i+1):2*(i+2)] - D[:,2*i:2*(i+1)]
            
            A1 = np.hstack((A1,A2))
            
        A1 = np.hstack((A1,D[:,len(D.T)-2:len(D.T)]))
        
    A1 = np.hstack((intercept,A1))
        
    return [ at , A1 ]

