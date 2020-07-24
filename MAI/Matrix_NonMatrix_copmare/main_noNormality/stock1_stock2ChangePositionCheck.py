# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:07:25 2020

@author: MAI
"""
count = 0
for i in range(len(table)):
    stock1 = table.stock1.iloc[i]
    stock2 = table.stock2.iloc[i]
    check = 0
    for j in range(len(table184)):
        if table184.stock1.iloc[j]==stock1 and table184.stock2.iloc[j]==stock2:
            count += 1
            check = 1
        elif table184.stock1.iloc[j]==stock2 and table184.stock2.iloc[j]==stock1:
            count += 1
            check = 1
    
    if check == 0:
        print('num:',i,'stock1:',stock1,'stock2:',stock2)
error1 = [56,74,169]
count = 0
for i in range(len(table184)):
    stock1 = table184.stock1.iloc[i]
    stock2 = table184.stock2.iloc[i]
    check = 0
    for j in range(len(table)):
        if table.stock1.iloc[j]==stock1 and table.stock2.iloc[j]==stock2:
            count += 1
            check = 1
        elif table.stock1.iloc[j]==stock2 and table.stock2.iloc[j]==stock1:
            count += 1
            check = 1
    
    if check == 0:
        print('num:',i,'stock1:',stock1,'stock2:',stock2)

error2 = [3,8,10,13,40,78,88]

print(table.iloc[error1,10:])
print(table184.iloc[error2,10:])


np.where(day1.columns=='6285')
np.where(day1.columns=='2474')


stock1 = day1.iloc[:150,130]
stock2 = day1.iloc[:150,71]

stock1_name = day1.columns.values[130]
stock2_name = day1.columns.values[71]

stock1 = day1.iloc[:150,71]
stock2 = day1.iloc[:150,130]

stock1_name = day1.columns.values[71]
stock2_name = day1.columns.values[130]
0.6373670152249097
0.06396523305638571