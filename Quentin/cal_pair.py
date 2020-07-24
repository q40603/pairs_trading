import pymysql
import os
import json

from formation_period import formation_period_single #, formation_period_pair
import accelerate_formation
import accelerate_trading
import ADF
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from sqlalchemy import create_engine

db_host = '140.113.24.2'
db_name = 'fintech'
db_user = 'fintech'
db_passwd = 'financefintech'
sqlEngine = create_engine('mysql+pymysql://'+db_user+':'+db_passwd+'@'+db_host+'/'+db_name, pool_recycle=3600)
dbConnection = sqlEngine.connect()


fin_db = pymysql.connect(
	host = db_host,
	user = db_user,
	password = db_passwd,
	db = db_name,

)
fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)



if __name__ == '__main__':
	choose_date = sys.argv[1]
	query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime >= '"+ choose_date +"' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
	fin_cursor.execute(query)
	result = fin_cursor.fetchall()
	fin_db.commit()
	df = pd.DataFrame(list(result))
	df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
	df = df.fillna(method='ffill')
	df = df.fillna(method='backfill')
	df.index = np.arange(0,len(df),1)

	unitroot_stock = ADF.adf.drop_stationary(ADF.adf(df))    
	a = accelerate_formation.pairs_trading(unitroot_stock)
	table = accelerate_formation.pairs_trading.formation_period( a )
	print(table)
	if not table.empty:
		table = table.drop(["model_type","skewness"],axis=1)
		table["f_date"] = datetime.strptime(choose_date,'%Y-%m-%d')
		table.to_sql("pairs", index=False,con = sqlEngine, if_exists = 'append', chunksize = 1000)
