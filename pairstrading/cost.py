#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:04:26 2019

@author: chaohsien
"""

#import pandas as pd
#import numpy as np

def tax( stock1_payoff , stock2_payoff , pos , tax_cost ):
    
    discount = 1
    
    #buy_cost = 1 + (0.1425/100) * discount                                    # 買券成本
    #sell_cost = 1 - (0.1425/100 + 0.3/100) * discount                         # 賣券所得
    #short_selling_cost = 1 - (0.1425/100 + 0.3/100 + 0.08/100) * discount     # 融券賣出所得
    #buy_short_cost = 1 + (0.1425/100) * discount                              # 回補融券成本
    
    buy_cost = 1                                                               # 買券成本
    sell_cost = 1 - tax_cost * discount                                        # 賣券所得
    short_selling_cost = 1 - tax_cost * discount                               # 融券賣出所得
    buy_short_cost = 1                                                         # 回補融券成本
    
    if pos == 1 or pos == -1:                                                  # 考慮開倉的交易成本
        
        if stock1_payoff < 0 and stock2_payoff < 0 :
            
            stock1_payoff = stock1_payoff * buy_cost
            stock2_payoff = stock2_payoff * buy_cost
            
        elif stock1_payoff > 0 and stock2_payoff < 0 :
            
            stock1_payoff = stock1_payoff * short_selling_cost
            stock2_payoff = stock2_payoff * buy_cost
            
        elif stock1_payoff < 0 and stock2_payoff > 0 :
            
            stock1_payoff = stock1_payoff * buy_cost
            stock2_payoff = stock2_payoff * short_selling_cost
            
        elif stock1_payoff > 0 and stock2_payoff > 0 :
            
            stock1_payoff = stock1_payoff * short_selling_cost
            stock2_payoff = stock2_payoff * short_selling_cost
        
    if pos == 0 or pos == -2:                                                               # 考慮平倉的交易成本
        
        if stock1_payoff < 0 and stock2_payoff < 0 :
            
            stock1_payoff = stock1_payoff * buy_short_cost
            stock2_payoff = stock2_payoff * buy_short_cost
            
        elif stock1_payoff > 0 and stock2_payoff < 0 :
            
            stock1_payoff = stock1_payoff * sell_cost
            stock2_payoff = stock2_payoff * buy_short_cost
            
        elif stock1_payoff < 0 and stock2_payoff > 0 :
            
            stock1_payoff = stock1_payoff * buy_short_cost
            stock2_payoff = stock2_payoff * sell_cost
            
        elif stock1_payoff > 0 and stock2_payoff > 0 :
            
            stock1_payoff = stock1_payoff * sell_cost
            stock2_payoff = stock2_payoff * sell_cost
            
    return [ stock1_payoff , stock2_payoff ]
            

def slip( price , payoff ):
    
    num = 0                                         # 設定滑價倍數
    
    if price < 10:
        
        if payoff > 0:                              # payoff為正，放空股票，所以便宜一個價格
            
            price = price - 0.01 * num
            
        else:                                       # payoff為負，買進股票，所以貴一個價格
            
            price = price + 0.01 * num
            
    elif price >= 10 and price < 50:
        
        if payoff > 0:                              # payoff為正，放空股票，所以便宜一個價格
            
            price = price - 0.05 * num
            
        else:                                       # payoff為負，買進股票，所以貴一個價格
            
            price = price + 0.05 * num
        
    elif price >= 50 and price < 100:
        
        if payoff > 0:                              # payoff為正，放空股票，所以便宜一個價格
            
            price = price - 0.1 * num
            
        else:                                       # payoff為負，買進股票，所以貴一個價格
            
            price = price + 0.1 * num
        
    elif price >= 100 and price < 500:
        
        if payoff > 0:                              # payoff為正，放空股票，所以便宜一個價格
            
            price = price - 0.5 * num
            
        else:                                       # payoff為負，買進股票，所以貴一個價格
            
            price = price + 0.5 * num
        
    elif price >= 500 and price < 1000:
        
        if payoff > 0:                              # payoff為正，放空股票，所以便宜一個價格
            
            price = price - 1 * num
            
        else:                                       # payoff為負，買進股票，所以貴一個價格
            
            price = price + 1 * num
        
    else:
        
        if payoff > 0:                              # payoff為正，放空股票，所以便宜一個價格
            
            price = price - 5 * num
            
        else:                                       # payoff為負，買進股票，所以貴一個價格
            
            price = price + 5 * num
        
    return price
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    