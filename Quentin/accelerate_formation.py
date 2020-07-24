# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:55:47 2019

@author: ChaoHsien
"""

#from formation_period import find_pairs
from formation_period import find_pairs
from MTSA import snr , zcr
from scipy.stats import skew

from multiprocessing import Process,Manager,Pool
import pandas as pd
import numpy as np
from vecm import para_vecm
from Matrix_function import order_select

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
    logy = y.copy()
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


class pairs_trading(object):

    def __init__(self , data):
		
        self.data = data 
        self.select_model = []
        self.weight = []
        self.name = []
        self.z_c_r = []
        self.s_n_r = []
        self.ave = []
        self.std = []
        self.ske = []
        self.stock1_name = []
        self.stock2_name = []

    def append_pairs_Result(self,result):
        self.weight.extend(result[0])
        self.name.extend(result[1])
        self.select_model.extend(result[2])
    '''
    def append_zcr_result(self,result):
        self.stock1_name.extend(result[0])
        self.stock1_name.extend(result[1])
        self.z_c_r.extend(result[2])
    '''
    def find_pairs(self):
		
        n = len(self.data.columns)-1
		
		# 平行找尋配對
        pool = Pool(processes=16) 
        for i in range(n):
            t = pool.apply_async(find_pairs, (i,n,self.data,),callback = self.append_pairs_Result)
        pool.close()
        pool.join()
	
		#self.data = data
        if not len(self.name):
            return 0
        self.name = pd.DataFrame(self.name) ; self.name.columns = ["stock1","stock2"]
        self.select_model = pd.DataFrame(self.select_model) ; self.select_model.columns = ["model_type"]
        #self.weight = pd.DataFrame(self.weight).drop([2],axis=1) ; self.weight.columns = ["w1","w2"]
        
        if len(pd.DataFrame(self.weight).T) == 2:
        
            self.weight = pd.DataFrame(self.weight) ; self.weight.columns = ["w1","w2"]
            
        else:
    
            self.weight = pd.DataFrame(self.weight).drop([2],axis=1) ; self.weight.columns = ["w1","w2"]
        return 1
         
    def form_table(self):


		# 將共整合係數標準化，此權重為資金權重，因此必須依股價高低轉為張數權重。
        for i in range(len(self.name)):
			
            total = abs(self.weight.w1[i]) + abs(self.weight.w2[i])
		
            self.weight.w2[i] = (self.weight.w2[i] / total )
            self.weight.w1[i] = (self.weight.w1[i] / total )
			
        table = pd.concat([self.name,self.select_model,self.weight],axis=1)  
		
		#print("共整合係數標準化 done in " + str(end - start))
		#------------------------------------------------------------------------------
		#計算spread序列，做單根檢定，並刪除非定態spread序列
		
        spread = np.zeros((len(self.data),len(table)))

        for i in range(len(table)):
            spread[:,i] = table.w1[i] * self.data[ table.stock1[i] ] + table.w2[i] * self.data[ table.stock2[i] ]

        self.spread = pd.DataFrame(spread)
		
		#print("刪除非定態spread序列 done in " + str(end - start))	
		#------------------------------------------------------------------------------
		# 計算信噪比 ( spread )------------------------------
		
        for i in range(len(self.spread.T)):
		
            y = self.spread.iloc[:,i]		
            self.s_n_r.append( snr(y,100) )
		
        self.s_n_r = pd.DataFrame(self.s_n_r) ; self.s_n_r.columns = ["snr"]
		
		#print("計算信噪比 done in " + str(end - start))	
		#------------------------------------------------------------------------------
		# 計算過零率 ( spread )------------------------------
		
        #pool = Pool(processes=16) 
        Boot = 500
        
        for j in range(len(self.spread.T)):
            y = self.spread.iloc[:,j]
            #t = pool.apply_async(zcr, (y,Boot,con.stock1[j],con.stock2[j],),callback=self.append_zcr_result)
            self.z_c_r.append(zcr(y,Boot))
        #pool.close()
        #pool.join()
        
        #self.stock1_name = pd.DataFrame(self.stock1_name) ; self.stock1_name.columns = ["stock1"]
        #self.stock2_name = pd.DataFrame(self.stock2_name) ; self.stock2_name.columns = ["stock2"]
        self.z_c_r = pd.DataFrame(self.z_c_r) ; self.z_c_r.columns = ["zcr"]
		
        #mix = pd.concat([self.stock1_name,self.stock2_name,self.z_c_r],axis=1)
        
		#print("計算過零率 done in " + str(end - start))	
		#------------------------------------------------------------------------------
		# 開倉門檻and平倉門檻and偏度--------------------------------------
		
        for i in range(len(self.spread.T)):
            
            y = self.spread.iloc[:,i]
            
            # 有時間趨勢項的模型必須分開計算
            if table.model_type[i] == 'model4':
                
                x = np.arange(0,len(y))
                b1 , b0 = np.polyfit(x,y,1)
                
                trend_line = x*b1 + b0
                y = y - trend_line          
                
		
            self.ave.append( np.mean(y) )
            self.std.append( np.std(y) )
            self.ske.append( skew(y) )

        self.ave = pd.DataFrame(self.ave) ; self.ave.columns = ["mu"]
        self.std = pd.DataFrame(self.std) ; self.std.columns = ["stdev"]
        self.ske = pd.DataFrame(self.ske) ; self.ske.columns = ["skewness"]
		#print("開倉門檻and平倉門檻 done in " + str(end - start))	
		#------------------------------------------------------------------------------
		# 整理表格
		#start = datetime.now()	
        
        #con = pd.concat([con,self.s_n_r],axis=1)
        
        #con = pd.merge( con , mix , on=["stock1","stock2"] , how="outer" )
        
        table = pd.concat([table,self.s_n_r,self.z_c_r,self.ave,self.std,self.ske],axis=1)
        print(table)
        
		#end = datetime.now()
        del self.s_n_r
        del self.z_c_r
        del self.ave
        del self.std
        del self.ske
        del self.select_model
        del self.weight
        del self.name

        stock1_name = table.stock1.astype('str',copy=False)
        stock2_name = table.stock2.astype('str',copy=False)
        test_stock1 = np.array(self.data[stock1_name].T)
        test_stock2 = np.array(self.data[stock2_name].T)
        mean = np.zeros(len(table))
        std = np.zeros(len(table))
        for i in range(len(table)):
            spread_m,spread = spread_mean(test_stock1,test_stock2,i,table)
            mean[i] = np.mean(spread_m[-1:])
            #std[i] = np.sqrt(np.mean(np.square(spread_m-spread)))
            std[i] = get_Estd(test_stock1,test_stock2,i,table)
        table['e_mu'] = mean
        table['e_stdev'] = std
        return table		

    def formation_period(self):
        
        if (self.find_pairs()):
            return self.form_table()
        else:
            return pd.DataFrame()
    
    def run(self):
        return self.formation_period()
    