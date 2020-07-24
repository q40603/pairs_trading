# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:53:01 2020

@author: MAI
"""
import numpy as np
from vecm import para_vecm
from scipy.stats import f , chi2

def Where_cross_threshold(trigger_spread, threshold, add_num):
    #initialize array
    check = np.zeros(trigger_spread.shape)
    #put on the condiction
    check[(trigger_spread - threshold) > 0] = add_num
    check[:,0] = check[:,1]
    #Open_trigger_array
    check = check[:,1:] - check[:,:-1]
    return check

def Where_threshold(trigger_spread, threshold, add_num, up):
    #initialize array
    check = np.zeros(trigger_spread.shape)
    #put on the condiction
    if up:
        check[(trigger_spread - threshold) > 0] = add_num
    else:
        check[(trigger_spread - threshold) < 0] = add_num
    check[:,0] = 0    
    return check

def tax(payoff,rate):
    tax_price = payoff * (1 - rate * (payoff > 0))
    return tax_price

def CNN_test(st1,st2,sp,v1,v2,tick,DetPos,table,NowOpen,model_CNN):
    if NowOpen:
        times = 1
    else:
        times = len(DetPos)
    #Array Initialize
    AllSprInput = []
    AllCharInput = []
    pair_pos = np.zeros([len(DetPos)],dtype = int)
    count = 0
    
    character = ['w1','w2','mu','stdev']
    TableChar = np.zeros([len(table),5])
    TableChar[:,:4] = np.array(table[character])

    if tick:
        s = 50
    else:
        s = 50
    for m in range(times):
        SprInput = np.zeros([100,5])
        CharInput = np.zeros([5])
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
            SprInput[:,3] = v1[pair,(s+index):(s+100+index)]
            SprInput[:,4] = v2[pair,(s+index):(s+100+index)]
            CharInput[4] = index/60
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
    mu = np.zeros([len(AllSprInput),1,5])
    mu[:,0,:2] = np.mean(AllSprInput[:,:,:2], axis=1)
    mu[:,0,2] = AllCharInput[:,2]
    #std
    stock_std = np.std(AllSprInput[:,:,:3], axis=1)
    std = np.ones([len(AllSprInput),1,5])
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

def VAR_model( y , p ):    
    k = len(y.T)     # 幾檔股票
    n = len(y)       # 資料長度
    
    xt = np.ones( ( n-p , (k*p)+1 ) )
    for i in range(n-p):
        a = 1
        for j in range(p):
            a = np.hstack( (a,y[i+p-j-1]) )
        a = a.reshape([1,(k*p)+1])
        xt[i] = a
    
    zt = np.delete(y,np.s_[0:p],axis=0)
    xt = np.mat(xt)
    zt = np.mat(zt)

    beta = ( xt.T * xt ).I * xt.T * zt                      # 計算VAR的參數
    
    A = zt - xt * beta                                      # 計算殘差
    sigma = ( (A.T) * A ) / (n-p)                           # 計算殘差的共變異數矩陣
        
    return [ sigma , beta ]

# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數--------------------------------------------------------------
def order_select( y , max_p ):
    
    k = len(y.T)     # 幾檔股票
    n = len(y)       # 資料長度
    
    bic = np.zeros((max_p,1))
    for p in range(1,max_p+1):
        sigma = VAR_model( y , p )[0]
        bic[p-1] = np.log( np.linalg.det(sigma) ) + np.log(n) * p * (k*k) / n
    bic_order = int(np.where(bic == np.min(bic))[0] + 1)        # 因為期數p從1開始，因此需要加1
    
    return bic_order

def fore_chow(stock1, stock2, model, Flen, give=False, p=0, A=0, ut=0, maxp=5):
    
    if model == 'model1':
        model_name = 'H2'
    elif model == 'model2':
        model_name = 'H1*'
    else:
        model_name = 'H1'
        
    day1 = ( np.vstack( [stock1, stock2] ).T )
    day1 = np.log(day1)
    h = len(day1) - Flen
    k = 2                                                                              # 幾檔股票
    n = Flen                                                                                # formation period 資料長度

    if give == False:
        y = ( np.vstack( [stock1[0:Flen], stock2[0:Flen]] ).T )            
        y = np.log(y)
        p = order_select(y,maxp)
        at , A, _ = para_vecm(y,model_name,p)    
#        at , A = para_vecm(y,model_name,p)    
        ut = np.dot(at,at.T)/len(at.T)    

    Remain_A = A.copy()
    Remain_ut = ut.copy()
    Remain_p = p
    
    A = A.T    
    phi_0 = np.eye(k)        
    A1 = np.delete(A,0,axis=0).T    
    phi = np.hstack( (np.zeros([k,2*(p-1)]) , phi_0) )
    sigma_t = np.dot( np.dot( phi_0 , ut ) , phi_0.T )                                         # sigma hat     
    ut_h = []

    for i in range(1,h+1):
        lag_mat = day1[ len(day1)-i-p-1 :  len(day1)-i , : ]    
        lag_mat = np.array(lag_mat[::-1])        
        if p == 1:
            ut_h.append( lag_mat[0].T - ( A[0].T + np.dot( A[1:k*p+1].T , lag_mat[1:2].T ) ).T )            
        else:
            ut_h.append( lag_mat[0].T - ( A[0].T + np.dot( A[1:k*p+1].T , lag_mat[1:k*p-1].reshape([k*p,1]) ) ).T )    

    for i in range(h-1):        
        a = phi[:,i*2:len(phi.T)]
        phi_i = np.dot( A1 , a.T )
        sigma_t = sigma_t + np.dot( np.dot( phi_i , ut ) , phi_i.T ) 
        phi = np.hstack( (phi , phi_i) )
    phi = phi[: , ((p-1)*k):len(phi.T)]
    ut_h = np.array(ut_h).reshape([1,h*2])
    e_t = np.dot( phi , ut_h.T )
    
    # 程式防呆，如果 sigma_t inverse 發散，則回傳有結構性斷裂。
    try:        
        tau_h = np.dot(np.dot( e_t.T , np.linalg.inv(sigma_t) ) , e_t ) / k        
    except:        
        return Remain_p, Remain_A, Remain_ut, 1    
    else:        
        if tau_h > float(f.ppf(0.99,k,n-k*p+1)):#tau_h > float(chi2.ppf(0.99,k)):    
            return Remain_p, Remain_A, Remain_ut, 1      # 有結構性斷裂
        else:    
            return Remain_p, Remain_A, Remain_ut, 0