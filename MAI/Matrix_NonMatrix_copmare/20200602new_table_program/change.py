# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:16:48 2020

@author: MAI
"""

import glob
import re
import pandas as pd

target = r"C:\Users\MAI\Desktop\trading_result\after20200204\trading_period_matrix_ver1.0\table\20200616testcorrect\*table.csv"
file_list=glob.glob(target)
regex=re.compile(r'.*\\20200616testcorrect(.*)')
for csv in file_list:
    try:
        new = pd.read_csv(csv)
        z = regex.match(csv)
        name = z.group(1)
    except:
        print(csv)
        continue
    else:
        if new.columns[0]=='Unnamed: 0':
            new = new.iloc[:,1:]
            new.to_csv('C:/Users/MAI/Desktop/trading_result/after20200204/trading_period_matrix_ver1.0/table/20200616testcorrect/{}'.format(name),index=False)
        