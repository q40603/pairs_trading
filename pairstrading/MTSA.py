# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 00:01:38 2018

@author: chuchu0936
"""

import numpy as np
import random
#import matlab
#import matlab.engine
from scipy.stats import f , chi2
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA

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

# structural break by chow test in formation period ------------------------------------------------------------------------
#eng = matlab.engine.start_matlab()

def chow_test( y , break_point , p , model , rank ):
    
    k = len(y.T)     # 幾檔股票
    n = len(y)       # 資料長度
    
    if model == 'H1':
        
        df1 = 1 + 2*k + k*rank + (k**2)
        df2 = 2 * (n-df1)
        
    elif model == 'H1*':
        
        df1 = 1 + k + k*rank + (k**2)
        df2 = 2 * (n-df1)
        
    else:
        
        df1 = k + k*rank + (k**2)
        df2 = 2 * (n-df1)
    
    # restrict model (full sample)
    #resi = np.array(eng.res_jci( matlab.double(y.tolist()) , model , (p-1) , rank ))
    
    rll = sum(sum(np.dot( resi.T , resi )))
    
    # unrestrict model (sub-sample)
    #resi_1 = np.array(eng.res_jci( matlab.double(y[0:break_point].tolist()) , model , (p-1) , rank ))
    #resi_2 = np.array(eng.res_jci( matlab.double(y[break_point:n].tolist()) , model , (p-1) , rank ))
    
    ull = sum(sum(np.dot( resi_1.T , resi_1 ))) + sum(sum(np.dot( resi_2.T , resi_2 )))
    
    #eng.quit()
    
    F = ( ( rll-ull )/df1 ) / ( ull/df2 )
    
    if F > f.ppf(0.95,df1,df2):
        
        return 1                # 拒絕虛無假設，表示有結構性斷裂
        
    else:
        
        return 0                # 不拒絕虛無假設，表示沒有結構性斷裂

# forecasting chow test in trading period ---------------------------------------------------------------------------------------

def fore_chow( stock1 , stock2 , stock1_trade , stock2_trade ):
    
    y = ( np.vstack( [stock1 , stock2] ).T )
    
    day1 = ( np.vstack( [stock1_trade , stock2_trade] ).T )
    
    k = len(y.T)                                                                              # 幾檔股票
    n = len(y)                                                                                # formation period 資料長度
    
    y = np.log(y)
    day1 = np.log(day1)

    h = len(day1) - n 
    
    p = order_select(y,5)                                                                     # 計算最佳落後期數
    ut , A = VAR_model(y , p)                                                                 # 計算VAR殘差共變異數與參數
    
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
        
        tau_h = np.dot(np.dot( e_t.T , np.linalg.inv(sigma_t) ) , e_t )
    
    except:
        
        return 1
    
    else:
        
        if tau_h > float(chi2.ppf(0.90,k)):
    
            return 1      # 有結構性斷裂

        else:
    
            return 0
	
# 計算信噪比(snr)---------------------------------------------------------------------------------------

def snr( y , lambda_hp ):
    
    n = len(y)
    H = np.hstack( [np.eye(n),np.zeros([n,2])] )
    Q = np.zeros([n,(n+2)])

    for i in range(n):
    
        Q[i,i] = 1
        Q[i,(i+1)] = -2
        Q[i,(i+2)] = 1

    A = np.dot(np.linalg.inv( np.dot(H.T , H) + lambda_hp * np.dot(Q.T,Q) ),H.T)

    y1 = y.iloc[::-1]                 # 將資料翻轉
    g = np.dot( A , y1 )[0:n][::-1]
    ct = y1.loc[::-1] - g

    signal_to_noise = ( np.var(ct) / np.var(g) )

    return signal_to_noise

# 計算過零率(zcr)-----------------------------------------------------------------------------------------
    
def zcr( y , B ):                     # B 為拔靴次數

    t = len(y)
    
    mu = np.mean(y)
    k = 1
    D = []
    for i in range(t-1):
    
        if ( y.iloc[i] - mu ) * ( y.iloc[i+1] - mu ) > 0 :
        
            k = k + 1 
        
        else:
        
            D.append(k)
            k = 1

    # 拔靴直到超過總交易分鐘數(271) , B次後平均 
    b = 0                       # 拔靴次數
    R = 0                       # 過零率拔靴B次後平均\
    while (b < B):
        N = 0                   # 累積次數 
        k = 0                   # 累積天數
        while True :
    
            k = k + D[random.randint(0,len(D)-1)]
            N = N + 1
    
            if k > t:
                break 
    
        b = b + 1

        R = R + ((N-1)/t)/B     # 拔靴後的過零率
        
    return R

def spread_chow( spread , i ):
    
    if i <= 25 or i >= 95 :
        
        return 0
    
    t = len(spread)
    
    try:    
        # full sample fitting ----------------------------------------------------------------------
    
        order = st.arma_order_select_ic(spread,max_ar=3,max_ma=3,ic='bic')  # 自動選取最適落後期數
        p , q = order.bic_min_order

        model = ARMA(spread, order=order.bic_min_order)
        result_arma = model.fit(disp=-1, method='css')

        residuals = result_arma.resid                                       # full sample residuals
        Sr = sum( residuals ** 2 )                                          # 殘差平方和

        # two sub sample fitting ------------------------------------------------------------------------

        sub_spread_1 = spread.loc[0:t-23]                                      # sub sample 1
        sub_spread_2 = spread.loc[t-22:len(spread)]                            # sub sample 2
    
        m1 = ARMA(sub_spread_1, order=order.bic_min_order)
        result_arma_1 = m1.fit(disp=-1, method='css')

        m2 = ARMA(sub_spread_2, order=order.bic_min_order)
        result_arma_2 = m2.fit(disp=-1, method='css')

        resi1 = result_arma_1.resid                                         # sub sample 1 residuals
        Sur1 = sum( resi1 ** 2 )                                            # 殘差平方和

        resi2 = result_arma_2.resid                                         # sub sample 2 residuals
        Sur2 = sum( resi2 ** 2 )                                            # 殘差平方和

        Sur = Sur1 + Sur2

        # 計算Chow test 檢定統計量--------------------------------------------------------------------------
        F = ( (Sr-Sur) / (p+q+1) ) / ( Sur / (len(spread)-2*(p+q+1)) )

        if F > f.ppf(0.95,(p+q+1),(len(spread)-2*(p+q+1))):
        
            return 1                # 拒絕虛無假設，表示有結構性斷裂
        
        else:
        
            return 0                # 不拒絕虛無假設，表示沒有結構性斷裂

    except:
        
        return 1

#spread = 0.5 * np.log(stock1) - 0.5 * np.log(stock2)

#t = 150

#lag = 23

#i = 12
#j = 30

#stock1 = day1.iloc[:,i]
#stock2 = day1.iloc[:,j]

#stock1_name = day1.columns.values[i]
#stock2_name = day1.columns.values[j]
        
#z = ( np.vstack( [stock1 , stock2] ).T )

#a = []
#for i in range(120):

    #a.append(fore_chow( z[0:149,:] , z[0:150+i,:] ))    # len(z[0:149,:]) - len(z[0:152,:])


#order_select(z[0:149,:],5)


