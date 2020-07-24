import pymysql
import os
import json
import csv
import time
import sys
import requests
from bs4 import BeautifulSoup
from .formation_period import formation_period_single #, formation_period_pair
from .trading_period import pairs
from datetime import datetime
from . import accelerate_formation
from . import accelerate_trading
from . import ADF
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



db_host = '140.113.24.2'
db_name = 'fintech'
db_user = 'fintech'
db_passwd = 'financefintech'

fin_db = pymysql.connect(
	host = db_host,
	user = db_user,
	password = db_passwd,
	db = db_name,

)
fin_cursor = fin_db.cursor(pymysql.cursors.DictCursor)



def get_stock_news(choose_date,cusip):
	date = choose_date
	date = date.replace("-","/")
	base_url = "https://tw.stock.yahoo.com/"
	res = requests.get('https://tw.stock.yahoo.com/q/h?s={}'.format(cusip))
	soup = BeautifulSoup(res.text,'html.parser')
	result = soup.find_all('tr',attrs={'bgcolor': '#fff1c4'})
	num = 0
	data = {}
	final = []
	for i in result:
		tmp = i.text.replace("\n","").replace("•","")
		print(tmp)
		tmp2 = num%2
		
		if not tmp2:
			data["title"] = tmp
			data["href"] = base_url + i.find('a')['href']
			print(base_url + i.find('a')['href'])
		if tmp2:
			data["time"] = tmp
			if(date in tmp):
				final.append(data)
			data = {}
			print("----------")
		num += 1
	print(final)
	return final 


def get_s_name(s1,s2):
	query = "select * from stock_name where s_id = {} or s_id = {}".format(s1, s2)
	fin_cursor.execute(query)
	result = fin_cursor.fetchall()
	return(result)


def get_pairs_spread(choose_date, s1, s2, w1, w2):
	fin_db.ping(reconnect = True)
	query1 = "select left(stime, 16) as mtimestamp, sum(volume * price)/sum(volume) as avg_price from " + s1 +  " where left(stime,10) = '" + choose_date + "' GROUP BY mtimestamp;" 
	fin_cursor.execute(query1)
	result1 = fin_cursor.fetchall()
	fin_db.commit()
	df = pd.DataFrame(list(result1))
	df['mtimestamp'] = pd.to_datetime(df['mtimestamp'])
	df = df.set_index('mtimestamp').resample('T')
	df = df.fillna(method='ffill')
	stock_1 = df.fillna(method='backfill')
	stock_1 = stock_1.reset_index()

	# for i,j in stock_1.iterrows():
	# 	print(j)


	query2 = "select left(stime, 16) as mtimestamp, sum(volume * price)/sum(volume) as avg_price from " + s2 +  " where left(stime,10) = '" + choose_date + "' GROUP BY mtimestamp;" 
	fin_cursor.execute(query2)
	result2 = fin_cursor.fetchall()
	fin_db.commit()
	df = pd.DataFrame(list(result2))
	df['mtimestamp'] = pd.to_datetime(df['mtimestamp'])
	df = df.set_index('mtimestamp').resample('T')
	df = df.fillna(method='ffill')
	stock_2 = df.fillna(method='backfill')
	stock_2 = stock_2.reset_index()
	spread = pd.DataFrame()
	spread["mtimestamp"] = stock_2["mtimestamp"]
	spread["avg_price"] = w1 * np.log(stock_1.avg_price) + w2 * np.log(stock_2.avg_price)
	spread = spread.fillna(method='ffill')
	spread = spread.fillna(method='backfill')

	return {
		"s1" : stock_1.to_dict("records"),
		"s2" : stock_2.to_dict("records"),
		"spread" : spread.to_dict("records")
	}
	# for i,j in stock_2.iterrows():
	# 	print(j)

def get_all_pairs(choose_date):
	fin_db.ping(reconnect = True)
	query = "select stock1, stock2, w1, w2, snr, zcr, mu, stdev from pairs where f_date = '" + choose_date + "' ;"
	fin_cursor.execute(query)
	data = fin_cursor.fetchall()
	fin_db.commit()
	return json.dumps(data)

def trade_all_pairs(choose_date, capital, maxi, open_time, stop_loss_time, tax_cost):
	fin_db.ping(reconnect = True)
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
		print(y)
		tmp = pairs( i , formate_time , y , min_data , tick_data , open_time , stop_loss_time , day1 , maxi , tax_cost , capital )
		print(tmp)


def trade_certain_pairs(choose_date, capital, maxi, open_time, stop_loss_time, tax_cost, pair_list):
	fin_db.ping(reconnect = True)
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
	# print(table)
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
	# print(tick_data)

	

	query = "select left(stime, 16) as mtimestamp, code , sum(volume * price)/sum(volume) as avg_price from s_price_tick where stime > '"+ choose_date +" 11:30' and stime <= '"+ choose_date +" 13:25' GROUP BY code, mtimestamp;"
	fin_cursor.execute(query)
	result = fin_cursor.fetchall()
	fin_db.commit()
	df = pd.DataFrame(list(result))
	df = df.pivot(index='mtimestamp', columns='code', values='avg_price')
	df = df.fillna(method='ffill')
	min_data = df.fillna(method='backfill')
	min_data.index = np.arange(0,len(min_data),1)
	# print(min_data)

	formate_time = 150

	# capital = 3000           # 每組配對資金300萬
	# maxi = 5                 # 股票最大持有張數
	# open_time = 1.5                 # 開倉門檻倍數
	# stop_loss_time = 10                  # 停損門檻倍數
	# tax_cost = 0
	l_table = len(table.index)
	for i in range(l_table):
		
		y = table.iloc[i,:]
		for j in pair_list:
			if (j[0] == y.stock1 and j[1] == y.stock2 ) or (j[0] == y.stock2 and j[1] == y.stock1 ):
				
				result = pairs( i , formate_time , y , min_data , tick_data , open_time , stop_loss_time , day1 , maxi , tax_cost , capital )
				return result





if __name__ == '__main__':
	choose_date = sys.argv[1]
	capital = 3000           # 每組配對資金300萬
	maxi = 5                 # 股票最大持有張數
	open_time = 1.5                 # 開倉門檻倍數
	stop_loss_time = 10                  # 停損門檻倍數
	tax_cost = 0
	pair_list = []
	#get_pairs_spread(choose_date, "s_2330", "s_2313")
	trade_all_pairs(choose_date, capital, maxi, open_time, stop_loss_time, tax_cost)



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
