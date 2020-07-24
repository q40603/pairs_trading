# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:55:47 2019

@author: ChaoHsien
"""

#from formation_period import find_pairs
from .formation_period import find_pairs
from .MTSA import snr , zcr
from scipy.stats import skew

from multiprocessing import Process,Manager,Pool
import pandas as pd
import numpy as np

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
        self.name = pd.DataFrame(self.name) ; self.name.columns = ["stock1","stock2"]
        self.select_model = pd.DataFrame(self.select_model) ; self.select_model.columns = ["model_type"]
        #self.weight = pd.DataFrame(self.weight).drop([2],axis=1) ; self.weight.columns = ["w1","w2"]
        
        if len(pd.DataFrame(self.weight).T) == 2:
        
            self.weight = pd.DataFrame(self.weight) ; self.weight.columns = ["w1","w2"]
            
        else:
    
            self.weight = pd.DataFrame(self.weight).drop([2],axis=1) ; self.weight.columns = ["w1","w2"]
    
         
    def form_table(self):

		# 將共整合係數標準化，此權重為資金權重，因此必須依股價高低轉為張數權重。
        for i in range(len(self.name)):
			
            total = abs(self.weight.w1[i]) + abs(self.weight.w2[i])
		
            self.weight.w2[i] = (self.weight.w2[i] / total )
            self.weight.w1[i] = (self.weight.w1[i] / total )
			
        con = pd.concat([self.name,self.select_model,self.weight],axis=1)  
		
		#print("共整合係數標準化 done in " + str(end - start))
		#------------------------------------------------------------------------------
		#計算spread序列，做單根檢定，並刪除非定態spread序列
		
        spread = np.zeros((len(self.data),len(con)))

        for i in range(len(con)):
            spread[:,i] = con.w1[i] * self.data[ con.stock1[i] ] + con.w2[i] * self.data[ con.stock2[i] ]

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
            if con.model_type[i] == 'model4':
                
                x = np.arange(0,len(y))
                b1 , b0 = np.polyfit(x,y,1)
                
                trend_line = x*b1 + b0
                y = y - trend_line          
                
                # 將spread消除趨勢項後，計算mu與std
                self.ave.append( np.mean(y) )
                self.std.append( np.std(y) )
                self.ske.append( skew(y) )
                               
            else:
		
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
        
        con = pd.concat([con,self.s_n_r,self.z_c_r,self.ave,self.std,self.ske],axis=1)
        
		#end = datetime.now()
        del self.s_n_r
        del self.z_c_r
        del self.ave
        del self.std
        del self.ske
        del self.select_model
        del self.weight
        del self.name
		#print("整理表格 done in " + str(end - start))	
        #print(con)
        return con			

    def formation_period(self):
        
        self.find_pairs()
        result = self.form_table()
            
        return result
    
    def run(self):
        return self.formation_period()
    