#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:33:32 2018

@author: chaohsien
"""

import pandas as pd
import numpy as np
#import matlab
#import matlab.engine
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.vector_ar.var_model import VARResults
from MTSA import order_select , snr , zcr #, chow_test 
from vecm import rank , eig , weigh

# 挑出定態的股票，並做回測------------------------------------------------------------------------------------------------------------------------
def formation_period_single(day1): 
    
    # 合併二日資料
    #min_price = np.log(pd.concat([day1,day2]))
    
    min_price = day1
    min_price = min_price.dropna(axis = 1)
    
    min_price.index  = np.arange(0,len(min_price),1)
    
    unit_stock = np.where(min_price.apply(lambda x: adfuller(x)[1] > 0.05 , axis = 0 ) == True )     # 找出單跟的股票
    min_price.drop(min_price.columns[unit_stock], axis=1 , inplace = True)                           # 刪除單跟股票，剩餘為定態序列。
    
    spread = min_price
    #------------------------------------------------------------------------------
    # 無母數開倉門檻--------------------------------------

    ave = []
    std = []
    for i in range(len(spread.T)):
    
        y = spread.iloc[:,i]
    
        ave.append( np.mean(y) )
        std.append( np.std(y) )
        
    ave = pd.DataFrame(ave) ; ave.columns = ["mu"]
    std = pd.DataFrame(std) ; std.columns = ["stdev"]
    
    # 程式防呆
    a = np.array(np.where( std < 0.0000001 ))
    
    if a.size > 1:
    
        a = int(np.delete(a,-1,axis=0))
        
        spread.drop(spread.columns[a],axis=1,inplace=True)
        ave.drop(ave.index[a],axis=0,inplace=True) ; ave.index = np.arange(0,len(ave),1)
        std.drop(std.index[a],axis=0,inplace=True) ; std.index = np.arange(0,len(std),1)
    
    #------------------------------------------------------------------------------
    # 計算過零率 ( spread )------------------------------

    Boot = 500
    z_c_r = []
    for j in range(len(spread.T)):
    
        y = spread.iloc[:,j]
    
        z_c_r.append( zcr(y,Boot) )

    z_c_r = pd.DataFrame(z_c_r) ; z_c_r.columns = ["zcr"]

    # -----------------------------------------------------------------------------
    stock_name = pd.DataFrame(spread.columns) ; stock_name.columns = ["stock"]
    
    con = pd.concat([stock_name,z_c_r,ave,std],axis=1)
    
    return con

# 挑出單根股票，帶入VECM中---------------------------------------------------------------------------------------------------------------------
def find_pairs( i , n , min_price):
    
    #eng=matlab.engine.start_matlab()  
    
    # 選擇適合的 VECM model，並且檢定 formation period 是否有結構性斷裂，並刪除該配對，其餘配對則回傳共整合係數。
    #rank = 1
    #t1 = int(len(min_price)*3/4)     # 一天的時間長度(偵測兩天中間是否有結構性斷裂)
    
    local_select_model = []
    local_weight = []
    local_name = []
    local_pval = []
    for j in range(i+1,n+1):
            
        stock1 = min_price.iloc[:,i]
        stock2 = min_price.iloc[:,j]
            
        stock1_name = min_price.columns.values[i]
        stock2_name = min_price.columns.values[j]
        
        z = ( np.vstack( [stock1 , stock2] ).T )
        model = VAR(z)
        p = order_select(z,5)
        #p = int(model.select_order(5).bic)
            
        # VAR 至少需落後1期
        if p < 1:     
                
            continue
            
        # portmanteau test
        if model.fit(p).test_whiteness( nlags = 3 ).pvalue < 0.05:
                
            continue
            
        # Normality test
        if model.fit(p).test_normality().pvalue < 0.05:
                
            continue
        
        #r1 = eng.rank_jci( matlab.double(z.tolist()) , 'H2' , (p-1) ) 
        #r2 = eng.rank_jci( matlab.double(z.tolist()) , 'H1*' , (p-1))
        #r3 = eng.rank_jci( matlab.double(z.tolist()) , 'H1' , (p-1) )
        
        r1 = rank( pd.DataFrame(z) , 'H2' , p ) 
        r2 = rank( pd.DataFrame(z) , 'H1*' , p )
        r3 = rank( pd.DataFrame(z) , 'H1' , p )
        r4 = rank( pd.DataFrame(z) , 'H*' , p )
            
        if r4 > 0:                          # 在 model 4 上有 rank
            
            if r3 > 0:                      # 在 model 3 上有 rank
                
                if r2 > 0:                  # 在 model 2 上有 rank
                    
                    if r1 > 0:              # select model 1 and model 2 and model 3 and model 4
                        
                        #lambda_model3 = eng.eig_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 )
                        #lambda_model4 = eng.eig_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r3 )
                
                        lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r3 )
                        lambda_model4 = eig( pd.DataFrame(z) , 'H*' , p , r3 )

                        test = np.log(lambda_model3/lambda_model4) * (len(min_price)-p)
                
                        if test > 3.8414:
                    
                            #bp1 = chow_test( z , t1 , p , 'H*' , r4 )
                    
                            #if bp1 == 0:               
                        
                            local_select_model.append('model4')
                    
                            #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r4 ) )
                            local_weight.append( weigh( pd.DataFrame(z) , 'H*' , p , r4 ) )
                    
                            local_name.append([stock1_name,stock2_name])
                   
                        else:
                            
                            #lambda_model2 = eng.eig_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 )
                    
                            lambda_model2 = eig( pd.DataFrame(z) , 'H1*' , p , r2 )
                    
                            test = np.log(lambda_model3/lambda_model2) * (len(min_price)-p)
                                
                            if test > 3.8414:
                        
                                #bp1 = chow_test( z , t1 , p , 'H1' , r3 )
                        
                                #if bp1 == 0:    
                            
                                local_select_model.append('model3')
                        
                                #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 ) )
                                local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                                
                                local_name.append([stock1_name,stock2_name])
                                
                            else:
                            
                                #lambda_model1 = eng.eig_jci( matlab.double(z.tolist()) , 'H2' , (p-1) , r1 )
                    
                                lambda_model1 = eig( pd.DataFrame(z) , 'H2' , p , r1 )
                    
                                test = np.log(lambda_model1/lambda_model2) * (len(min_price)-p)
                                
                                if test > 3.8414:
                        
                                    #bp1 = chow_test( z , t1 , p , 'H1*' , r2 )
                        
                                    #if bp1 == 0:    
                            
                                    local_select_model.append('model2')
                        
                                    #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 ) )
                                    local_weight.append( weigh( pd.DataFrame(z) , 'H1*' , p , r2 ) )
                                
                                    local_name.append([stock1_name,stock2_name])
                                
                                else:
                            
                                    #bp1 = chow_test( z , t1 , p , 'H2' , r1 ) 
                    
                                    #if bp1 == 0:      
                        
                                    local_select_model.append('model1')
                    
                                    #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H2' , (p-1) , r1 ) )
                                    local_weight.append( weigh( pd.DataFrame(z) , 'H2' , p , r1 ) )
                    
                                    local_name.append([stock1_name,stock2_name])
                        
                        
                    else:                   # select model 2 and model 3 and model 4
                   
                        #lambda_model3 = eng.eig_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 )
                        #lambda_model4 = eng.eig_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r3 )
                
                        lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r3 )
                        lambda_model4 = eig( pd.DataFrame(z) , 'H*' , p , r3 )

                        test = np.log(lambda_model3/lambda_model4) * (len(min_price)-p)
                
                        if test > 3.8414:
                    
                            #bp1 = chow_test( z , t1 , p , 'H*' , r4 )
                    
                            #if bp1 == 0:               
                        
                            local_select_model.append('model4')
                    
                            #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r4 ) )
                            local_weight.append( weigh( pd.DataFrame(z) , 'H*' , p , r4 ) )
                    
                            local_name.append([stock1_name,stock2_name])
                   
                        else:
                            
                            #lambda_model2 = eng.eig_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 )
                    
                            lambda_model2 = eig( pd.DataFrame(z) , 'H1*' , p , r2 )
                    
                            test = np.log(lambda_model3/lambda_model2) * (len(min_price)-p)
                                
                            if test > 3.8414:
                        
                                #bp1 = chow_test( z , t1 , p , 'H1' , r3 )
                        
                                #if bp1 == 0:    
                            
                                local_select_model.append('model3')
                        
                                #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 ) )
                                local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                                
                                local_name.append([stock1_name,stock2_name])
                                
                            else:
                            
                                #bp1 = chow_test( z , t1 , p , 'H1*' , r2 ) 
                    
                                #if bp1 == 0:      
                        
                                local_select_model.append('model2')
                    
                                #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1*' , (p-1) , r2 ) )
                                local_weight.append( weigh( pd.DataFrame(z) , 'H1*' , p , r2 ) )
                    
                                local_name.append([stock1_name,stock2_name])
                            
                else:                  # select model3 and model4
                    
                    #lambda_model3 = eng.eig_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 )
                    #lambda_model4 = eng.eig_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r3 )
                
                    lambda_model3 = eig( pd.DataFrame(z) , 'H1' , p , r3 )
                    lambda_model4 = eig( pd.DataFrame(z) , 'H*' , p , r3 )

                    test = np.log(lambda_model3/lambda_model4) * (len(min_price)-p)
                
                    if test > 3.8414:
                            
                        #bp1 = chow_test( z , t1 , p , 'H*' , r4 )
                    
                        #if bp1 == 0:               
                        
                        local_select_model.append('model4')
                    
                        #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r4 ) )
                        local_weight.append( weigh( pd.DataFrame(z) , 'H*' , p , r4 ) )
                    
                        local_name.append([stock1_name,stock2_name])
                   
                    else:
                         
                        #bp1 = chow_test( z , t1 , p , 'H1' , r3 ) 
                    
                        #if bp1 == 0:      
                        
                        local_select_model.append('model3')
                        
                        #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H1' , (p-1) , r3 ) )
                        local_weight.append( weigh( pd.DataFrame(z) , 'H1' , p , r3 ) )
                        
                        local_name.append([stock1_name,stock2_name])
                        
                    
            else :                     # 只在 model 4 上有rank
                    
                    
                #bp1 = chow_test( z , t1 , p , 'H1' , r4 ) 
                
                #if bp1 == 0:            
                    
                local_select_model.append('model4')
                
                #weight.append( eng.coin_jci( matlab.double(z.tolist()) , 'H*' , (p-1) , r4 ) )
                local_weight.append( weigh( pd.DataFrame(z) , 'H*' , p , r4 ) )
                
                local_name.append([stock1_name,stock2_name])
                
                
        else:       # 表示此配對無rank
            
            continue
        
        local_pval.append(1)
        
    return local_weight, local_name, local_select_model, local_pval

