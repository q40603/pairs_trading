import pandas as pd
import numpy as np
import sys
from datetime import datetime
def min_weight_avg(file_name=None, f_start= None, f_end = None):
	with open( './top100.txt' , 'r' ) as f:
		top100 = f.read().splitlines()
	#variable declare, explain later

	last_time = datetime.strptime(str(f_end),'%H%M') if f_end != None else datetime.strptime('1129','%H%M')
	f_start_time = datetime.strptime(str(f_start),'%H%M') if f_end != None else datetime.strptime('0900','%H%M')

	#load daily data
	data = pd.read_csv( "f:/stock/" + file_name+"_stock.csv", usecols = ['商品名稱','時間', '狀態註記','成交價','成交量'] ,dtype={'時間':'object'}, encoding="big5")
	#delte rows with volume 0 and before 9:00 after 13:25
	data = data[(data['狀態註記'] == 0)  & (data['成交量'] != 0) & (data['商品名稱'].isin(top100))]

	#format time to datetime format
	data['時間'] = data['時間'].apply(lambda x: datetime.strptime(x[0:4],'%H%M'))
	data = data[(data['時間'] <= last_time)  & (data['時間'] >= f_start_time) ]
	
	#select the top 100 frequently traded stock and  dump to list
	#tmp = data.groupby("商品名稱")['成交量'].sum().nlargest(100)
	
	#top100 = tmp.index.tolist()
	#delte stock not in top100 list
	#data = data[data['商品名稱'].isin(top100)]
	#construct stock pool
	pool = dict()
	#calculate weight average
	tmp = data.groupby(["商品名稱","時間"]).apply(lambda x: (x['成交價']/100 * x['成交量']).sum() / x["成交量"].sum())
	prev_cusip = ""
	prev_time = datetime.strptime(str(f_end),'%H%M') if f_end != None else datetime.strptime('1129','%H%M')
	prev_price = 0
	#loop over the pandas series
	#fill the price in the hole (9:23,9:25) -> fill 9:25's price into 9:24
	for i,r in tmp.items():
		#stock is the same as prevois loop
	    if(prev_cusip == i[0]):
	    	# fill the gap is necessary
	    	gap = (i[1] - prev_time).seconds
	    	for j in range(int(gap/60)):
	    		pool[i[0]].append(float(format(r, '.5f')))
	    #stock is different from prevois loop
	    else:
	    	# construct new array
	    	pool[i[0]] = [float(format(r, '.5f'))]
	    	#calculate gap between previos stock last transaction time and end_time(13:24)
	    	gap = (last_time - prev_time).seconds
	    	#fill previos
	    	for j in range(int(gap/60)):
	    		pool[prev_cusip].append(float(format(prev_price, '.5f')))
	    	#calculate gap between current stock first transaction time and f_start_time(9:00)
	    	gap = (i[1] - f_start_time).seconds
	    	#fill previos
	    	for j in range(int(gap/60)):
	    		pool[i[0]].append(float(format(r, '.5f')))
	    #store current data
	    prev_cusip = i[0]
	    prev_time = i[1]
	    prev_price = r
	# fill the gap if last stock's last transaction time is before 13:24
	gap = (last_time - prev_time).seconds
	for j in range(int(gap/60)):
	    pool[prev_cusip].append(float(format(prev_price, '.5f')))

	tmp = {num: len(pool[num]) for num in pool}
	#bug still needs to be fixed
	#print(pool)
	#return pandas dataframe
	return (pd.DataFrame.from_dict(pool))


def sec_weight_avg(file_name=None, t_start= None, t_end = None):
	with open( './top100.txt' , 'r' ) as f:
		top100 = f.read().splitlines()
	#variable declare, explain later

	last_time = datetime.strptime(str(t_end),'%H%M') if t_end != None else datetime.strptime('132459','%H%M%S')
	t_start_time = datetime.strptime(str(t_start),'%H%M') if t_end != None else datetime.strptime('113000','%H%M%S')

	#load daily data
	data = pd.read_csv( file_name+"_stock.csv", usecols = ['商品名稱','時間', '狀態註記','成交價','成交量'] ,dtype={'時間':'object'}, encoding="big5")
	#delte rows with volume 0 and before 9:00 after 13:25
	data = data[(data['狀態註記'] == 0)  & (data['成交量'] != 0) & (data['商品名稱'].isin(top100))]


	#format time to datetime format
	data['時間'] = data['時間'].apply(lambda x: datetime.strptime(x[0:6],'%H%M%S'))
	data = data[(data['時間'] <= last_time)  & (data['時間'] >= t_start_time) ]
	
	#select the top 100 frequently traded stock and  dump to list
	#tmp = data.groupby("商品名稱")['成交量'].sum().nlargest(100)
	
	#top100 = tmp.index.tolist()
	#delte stock not in top100 list
	#data = data[data['商品名稱'].isin(top100)]
	#construct stock pool
	pool = dict()
	#calculate weight average
	tmp = data.groupby(["商品名稱","時間"]).apply(lambda x: (x['成交價']/100 * x['成交量']).sum() / x["成交量"].sum())
	prev_cusip = ""
	prev_time = datetime.strptime(str(t_end),'%H%M%S') if t_end != None else datetime.strptime('132459','%H%M%s')
	prev_price = 0
	#loop over the pandas series
	#fill the price in the hole (9:23,9:25) -> fill 9:25's price into 9:24
	for i,r in tmp.items():
		#stock is the same as prevois loop
	    if(prev_cusip == i[0]):
	    	# fill the gap is necessary
	    	gap = (i[1] - prev_time).seconds
	    	for j in range(int(gap)):
	    		pool[i[0]].append(float(format(r, '.5f')))
	    #stock is different from prevois loop
	    else:
	    	# construct new array
	    	pool[i[0]] = [float(format(r, '.5f'))]
	    	#calculate gap between previos stock last transaction time and end_time(13:24)
	    	gap = (last_time - prev_time).seconds
	    	#fill previos
	    	if(prev_cusip != ""):
		    	for j in range(int(gap)):
		    		pool[prev_cusip].append(float(format(prev_price, '.5f')))
	    	#calculate gap between current stock first transaction time and t_start_time(9:00)
	    	gap = (i[1] - t_start_time).seconds
	    	#fill previos
	    	for j in range(int(gap)):
	    		pool[i[0]].append(float(format(r, '.5f')))
	    #store current data
	    prev_cusip = i[0]
	    prev_time = i[1]
	    prev_price = r
	# fill the gap if last stock's last transaction time is before 13:24
	gap = (last_time - prev_time).seconds
	for j in range(int(gap)):
	    pool[prev_cusip].append(float(format(prev_price, '.5f')))

	tmp = {num: len(pool[num]) for num in pool}
	#bug still needs to be fixed
	#print(pool)
	#return pandas dataframe
	return (pd.DataFrame.from_dict(pool))


if __name__ == '__main__':
	data = sec_weight_avg(file_name="20190311", t_start = '1130', t_end = '1324')
	print(data)
