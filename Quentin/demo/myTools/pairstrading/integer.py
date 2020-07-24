#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:49:12 2019

@author: chaohsien
"""

import numpy as np

# 最小化整數比
def min_integer( w1 , w2 , stock1_max , stock2_max ):

    y = abs(w2 / w1)
    theta = np.arctan(y)

    sq = []
    for i in range(1,stock1_max+1):

        for j in range(1,stock2_max+1):
        
            sq.append( [ i , j , abs(theta - np.arctan(j/i)) ]) 
    
    sq = np.array(sq)

    min_pos = np.array(np.where(sq[:,2] == np.min(sq[:,2])) )  # 挑出角度差最小的權重

    if len(min_pos.T) > 1:                                     # 如果有重複，則挑第一個
    
        min_pos = min_pos[0,0]

    else:
    
        min_pos = int(min_pos)
        
    #回傳值依原始正負調整-------------------------------------------------------------------------------
    
    if w1 > 0 and w2 > 0:

        w1 = sq[min_pos,0]
        w2 = sq[min_pos,1]
        
    elif w1 < 0 and w2 > 0:
        
        w1 = -sq[min_pos,0]
        w2 =  sq[min_pos,1]
        
    elif w1 > 0 and w2 < 0:
        
        w1 =  sq[min_pos,0]
        w2 = -sq[min_pos,1]
        
    else:
        
        w1 = -sq[min_pos,0]
        w2 = -sq[min_pos,1]
        
    return [ w1 , w2 ]
        
#將資金權重換成股票張數權重，並進行整數化 ; maxi為最大張數。
def num_weight( w1 , w2 , price1 , price2 , maxi , initial_capital):
    
    #initial_capital = 3000      # 總資產300萬台幣
    
    stock1_num = (w1 * initial_capital)/price1
    stock2_num = (w2 * initial_capital)/price2
    
    if abs(stock1_num) > maxi or abs(stock2_num) > maxi :
        
        stock1_maxi = maxi
        stock2_maxi = maxi
        
    elif abs(stock1_num) > maxi or abs(stock2_num) < maxi :
        
        stock1_maxi = maxi
        stock2_maxi = int(round(stock2_num))
        
    elif abs(stock1_num) < maxi or abs(stock2_num) > maxi :
        
        stock1_maxi = int(round(stock1_num))
        stock2_maxi = maxi
        
    else:
        
        stock1_maxi = int(round(stock1_num))
        stock2_maxi = int(round(stock2_num))
    
    w1 , w2 = min_integer( stock1_num , stock2_num , stock1_maxi , stock2_maxi )
    
    return [ w1 , w2 ]

#num_weight( -0.440871 , 0.559129 , 299.5 , 13.5  , 100 )
