import os
import json
import csv
import time
import sys
import re
from datetime import datetime
from formation_period import formation_period_single #, formation_period_pair
from trading_period import pairs
from datetime import datetime
import accelerate_formation
import accelerate_trading
import ADF
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# db_host = '140.113.24.2'
# db_name = 'fintech'
# db_user = 'fintech'
# db_passwd = 'financefintech'

# fin_db = pymysql.connect(
# 	host = db_host,
# 	user = db_user,
# 	password = db_passwd,
# 	db = db_name,

# )
# fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)

base = sys.argv[1]

def trade_all_pairs(capital, maxi, open_time, stop_loss_time, tax_cost):

	f_50 = open('50_corp.txt', encoding='utf-8')
	top_50 = [i.split(",")[0] for i in f_50]
	header_name = ["mtimestamp", "code", "price", "vol", "acu_vol"]
	print(top_50)
	dirs = os.listdir(base)
	dirs = ''.join(dirs)
	all_date = re.findall(r"\d{8,8}",dirs)
	for date in all_date:
		print(date)
		if(int(date)<20181027):
			continue
		
		data = pd.read_csv("".join([base,date,"_Match.txt"]), header=None,usecols = [0,1,3,4,5], names=header_name, sep=',', dtype={'mtimestamp':'object','code': 'object'}, encoding="utf-8", engine="python", error_bad_lines=False)
		data = data[data['code'].isin(top_50)]
		

		data["mtimestamp"] = data["mtimestamp"].apply(lambda x: datetime.strptime(" ".join([date,x[0:4]]),'%Y%m%d %H%M'))
		#print(data)
		tick_data = data.copy()

		data = data.groupby(["mtimestamp","code"]).apply(lambda x: (x['price'] * x['vol']).sum() / x["vol"].sum())
		data = data.unstack().resample('T').ffill().bfill()
		min_data = data.copy()

		day1 = data.reset_index()


		# print(day1)
		day1.index = np.arange(0,len(day1),1)
		# print(day1)
		day1_1 = day1.iloc[0 : 149,:]
		day1_1 = day1_1.drop(columns=['mtimestamp' ])
		print(day1_1)
		# print(df)
		day1_1.index = np.arange(0,len(day1_1),1)
		
		unitroot_stock = ADF.adf.drop_stationary(ADF.adf(day1_1.select_dtypes(exclude=['object'])))    
		a = accelerate_formation.pairs_trading(unitroot_stock)
		table = accelerate_formation.pairs_trading.formation_period( a )
		if table.empty:
			continue
			# print(table)
		print(table)
		# tracking_list = pd.concat([table.stock1, table.stock2])
		# tracking_list = pd.unique(tracking_list)
		# print(tracking_list)


		
	#========================================== back test ==============================================



		#query = "select left(stime, 16) as mtimestamp, code , price from s_price_tick where stime >= '"+ choose_date +" 11:29' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"   
		# fin_cursor.execute(query)
		# result = fin_cursor.fetchall()
		# fin_db.commit()
		# df = pd.DataFrame(list(result))
		# df = df.pivot(index='mtimestamp', columns='code', values='price')
		# df = df.fillna(method='ffill')
		# tick_data = df.fillna(method='backfill')
		# tick_data.index = np.arange(0,len(tick_data),1)
		last_time = datetime.strptime(date+' 1325','%Y%m%d %H%M')
		start_time = datetime.strptime(date+' 1129','%Y%m%d %H%M')	

		
		tick_data = tick_data.drop(["vol","acu_vol"],axis=1)

		tick_data = tick_data.groupby(["code","mtimestamp"]).tail(1)
		#print(tick_data)
		tick_data = tick_data.pivot(index='mtimestamp', columns='code', values='price')
		tick_data = tick_data.ffill().bfill()
		print(tick_data)
		tick_data = tick_data.iloc[150:265]
		#tick_data = tick_data[(tick_data["mtimestamp"]>=start_time & tick_data["mtimestamp"]<=last_time)]
		print(tick_data)
		tick_data.index = np.arange(0,len(tick_data),1)		
		

		

		# query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime > '"+ choose_date +" 11:30' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
		# fin_cursor.execute(query)
		# result = fin_cursor.fetchall()
		# fin_db.commit()
		# df = pd.DataFrame(list(result))
		# df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
		# df = df.fillna(method='ffill')
		# start_time = datetime.strptime(date+' 1130','%Y%m%d %H%M')
		# min_data = min_data[(min_data["mtimestamp"]>start_time & min_data["mtimestamp"]<=last_time)]
		# min_data.index = np.arange(0,len(min_data),1)
		# print(min_data)
		min_data = day1.iloc[150:265]
		min_data.index = np.arange(0,len(min_data),1)
		formate_time = 150

		# capital = 3000           # 每組配對資金300萬
		# maxi = 5                 # 股票最大持有張數
		# open_time = 1.5                 # 開倉門檻倍數
		# stop_loss_time = 10                  # 停損門檻倍數
		# tax_cost = 0
		l_table = len(table.index)
		for i in range(l_table):
			print(i)
			y = table.iloc[i,:]
			print(y)
			tmp = pairs(i , formate_time , y , min_data , tick_data , open_time , stop_loss_time , day1 , maxi , tax_cost , capital )
				
			print(tmp)


def trade_certain_pairs(choose_date, capital, maxi, open_time, stop_loss_time, tax_cost, pair_list):

		query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime >= '"+ choose_date +"' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
		print(query)
		fin_cursor.execute(query)
		result = fin_cursor.fetchall()
		fin_db.commit()
		df = pd.DataFrame(list(result))
		df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
		df = df.fillna(method='ffill')
		day1 = df.fillna(method='backfill')
		day1 = day1.reset_index()
		# print(day1)
		day1.index = np.arange(0,len(day1),1)
		# print(day1)
		day1_1 = day1.iloc[0 : 149,:]
		# print(df)
		day1_1.index = np.arange(0,len(day1_1),1)
		# print(len(day1_1.index))

		# print(df)

		query = "select distinct f_date from pairs where f_date = '"+ choose_date +"';"
		fin_cursor.execute(query)
		result = list(fin_cursor.fetchall())
		fin_db.commit()
		if (not len(result)):
			unitroot_stock = ADF.adf.drop_stationary(ADF.adf(day1_1.select_dtypes(exclude=['object'])))    
			a = accelerate_formation.pairs_trading(unitroot_stock)
			table = accelerate_formation.pairs_trading.formation_period( a )
			for i, j in table.iterrows():
				sql = "INSERT INTO pairs (stock1, stock2, w1, w2, snr, zcr, mu, stdev, f_date ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
				try :
					fin_cursor.execute(sql, (str(j["stock1"]), str(j["stock2"]), str(j["w1"]), str(j["w2"]), str(j["snr"]), str(j["zcr"]), str(j["mu"]), str(j["stdev"]), str(choose_date)))
				except Exception as e:
					print(e,j)
			fin_db.commit()
			# print(table)
		else:
			print(datetime.now().strftime("%Y-%b-%d"))
			query = "select * from pairs where f_date = '"+ choose_date +"';"
			fin_cursor.execute(query)
			result = fin_cursor.fetchall()
			fin_db.commit()
			table = pd.DataFrame(list(result))
			table.index = np.arange(0,len(table),1)
			# print(table)
		print(table)
		# tracking_list = pd.concat([table.stock1, table.stock2])
		# tracking_list = pd.unique(tracking_list)
		# print(tracking_list)


		
	#========================================== back test ==============================================



		query = "select left(stime, 16) as mtimestamp, code , price from s_price_tick where stime >= '"+ choose_date +" 11:29' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"   
		fin_cursor.execute(query)
		result = fin_cursor.fetchall()
		fin_db.commit()
		df = pd.DataFrame(list(result))
		df = df.pivot(index='mtimestamp', columns='code', values='price')
		df = df.fillna(method='ffill')
		tick_data = df.fillna(method='backfill')
		tick_data.index = np.arange(0,len(tick_data),1)
		print(tick_data)

		

		query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime > '"+ choose_date +" 11:30' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
		fin_cursor.execute(query)
		result = fin_cursor.fetchall()
		fin_db.commit()
		df = pd.DataFrame(list(result))
		df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
		df = df.fillna(method='ffill')
		min_data = df.fillna(method='backfill')
		min_data.index = np.arange(0,len(min_data),1)
		print(min_data)

		formate_time = 150

		# capital = 3000           # 每組配對資金300萬
		# maxi = 5                 # 股票最大持有張數
		# open_time = 1.5                 # 開倉門檻倍數
		# stop_loss_time = 10                  # 停損門檻倍數
		# tax_cost = 0
		l_table = len(table.index)
		for i in range(l_table):
			print(i)
			y = table.iloc[i,:]
			for j in pair_list:
				if (j[0] == y.stock1 and j[1] == y.stock2 ) or (j[0] == y.stock2 and j[1] == y.stock1 ):
					print(y)
					tmp = pairs( i , formate_time , y , min_data , tick_data , open_time , stop_loss_time , day1 , maxi , tax_cost , capital )
					print(tmp)
					break


if __name__ == '__main__':
	#choose_date = sys.argv[1]
	capital = 3000           # 每組配對資金300萬
	maxi = 5                 # 股票最大持有張數
	open_time = 1.5                 # 開倉門檻倍數
	stop_loss_time = 10                  # 停損門檻倍數
	tax_cost = 0
	trade_all_pairs(capital, maxi, open_time, stop_loss_time, tax_cost)



# 	query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime >= '"+ choose_date +"' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
# 	print(query)
# 	fin_cursor.execute(query)
# 	result = fin_cursor.fetchall()
# 	fin_db.commit()
# 	df = pd.DataFrame(list(result))
# 	df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
# 	df = df.fillna(method='ffill')
# 	day1 = df.fillna(method='backfill')
# 	day1 = day1.reset_index()
# 	# print(day1)
# 	day1.index = np.arange(0,len(day1),1)
# 	# print(day1)
# 	day1_1 = day1.iloc[0 : 149,:]
# 	# print(df)
# 	day1_1.index = np.arange(0,len(day1_1),1)
# 	# print(len(day1_1.index))

# 	# print(df)

# 	query = "select distinct f_date from pairs where f_date = '"+ choose_date +"';"
# 	fin_cursor.execute(query)
# 	result = list(fin_cursor.fetchall())
# 	fin_db.commit()
# 	if (not len(result)):
# 		unitroot_stock = ADF.adf.drop_stationary(ADF.adf(day1_1.select_dtypes(exclude=['object'])))    
# 		a = accelerate_formation.pairs_trading(unitroot_stock)
# 		table = accelerate_formation.pairs_trading.formation_period( a )
# 		for i, j in table.iterrows():
# 			sql = "INSERT INTO pairs (stock1, stock2, w1, w2, snr, zcr, mu, stdev, f_date ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
# 			try :
# 				fin_cursor.execute(sql, (str(j["stock1"]), str(j["stock2"]), str(j["w1"]), str(j["w2"]), str(j["snr"]), str(j["zcr"]), str(j["mu"]), str(j["stdev"]), str(choose_date)))
# 			except Exception as e:
# 				print(e,j)
# 		fin_db.commit()
# 		# print(table)
# 	else:
# 		print(datetime.now().strftime("%Y-%b-%d"))
# 		query = "select * from pairs where f_date = '"+ choose_date +"';"
# 		fin_cursor.execute(query)
# 		result = fin_cursor.fetchall()
# 		fin_db.commit()
# 		table = pd.DataFrame(list(result))
# 		table.index = np.arange(0,len(table),1)
# 		# print(table)
# 	print(table)
# 	# tracking_list = pd.concat([table.stock1, table.stock2])
# 	# tracking_list = pd.unique(tracking_list)
# 	# print(tracking_list)


	
# #========================================== back test ==============================================



# 	query = "select left(stime, 16) as mtimestamp, code , price from s_price_tick where stime >= '"+ choose_date +" 11:29' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"   
# 	fin_cursor.execute(query)
# 	result = fin_cursor.fetchall()
# 	fin_db.commit()
# 	df = pd.DataFrame(list(result))
# 	df = df.pivot(index='mtimestamp', columns='code', values='price')
# 	df = df.fillna(method='ffill')
# 	tick_data = df.fillna(method='backfill')
# 	tick_data.index = np.arange(0,len(tick_data),1)
# 	print(tick_data)

	

# 	query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime > '"+ choose_date +" 11:30' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
# 	fin_cursor.execute(query)
# 	result = fin_cursor.fetchall()
# 	fin_db.commit()
# 	df = pd.DataFrame(list(result))
# 	df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
# 	df = df.fillna(method='ffill')
# 	min_data = df.fillna(method='backfill')
# 	min_data.index = np.arange(0,len(min_data),1)
# 	print(min_data)

# 	formate_time = 150

# 	# capital = 3000           # 每組配對資金300萬
# 	# maxi = 5                 # 股票最大持有張數
# 	# open_time = 1.5                 # 開倉門檻倍數
# 	# stop_loss_time = 10                  # 停損門檻倍數
# 	# tax_cost = 0
# 	l_table = len(table.index)
# 	for i in range(l_table):
# 		print(i)
# 		y = table.iloc[i,:]
# 		print(y)
# 		tmp = pairs( i , formate_time , y , min_data , tick_data , open_time , stop_loss_time , day1 , maxi , tax_cost , capital )
# 		print(tmp)






	# prev_time = 0
	# with open('20190702_stock.csv','r', encoding="big5") as myFile:
	# 	lines=csv.reader(myFile)
	# 	for line in lines:
	# 		if (line[0] in tracking_list) and (int(line[6][0:4]) >= 1130):
	# 			if int(line[6][0:4]) != prev_time:
	# 				print(prev_time)
	# 				prev_time = int(line[6][0:4])
	# 			print(line)
			# time.sleep(0.01)
			# n = n - 1
			# if n == 0:
			# 	time.sleep(0.1)
			# 	n = 50
	
	# for i,r in df.iterrows():
	# 	print(r)
