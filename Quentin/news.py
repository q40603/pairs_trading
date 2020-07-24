from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import pandas as pd
import requests
import sys
import json
from datetime import datetime, timedelta
from copy import copy
import time

r = requests.session()
client = MongoClient("mongodb://localhost:27017/")
db = client["cfda"]
all_db = {}

f = open("./0050.txt",encoding="utf-8")
for i in f.readlines():
    i = i[:4]
    print(i)
    c_id = i 
    try:
        news = r.get("http://clip2.cs.nccu.edu.tw/~pcchien/idv_crawl.py?src=ctee,yahoo,udn&stock={}".format(c_id))
        news = json.loads(news.text)
        print(c_id, len(news))
        time.sleep(3)
    except Exception as e:
        print(e)
        continue
    new = []
    c_id = "c_" + c_id
    for j in news:
        j["time"] = pd.Timestamp(j['time'])
        if not db[c_id].find({'url': { "$in": [j["url"]]}}).count():
            print("append", j)
            db[c_id].insert(copy(j))
            new.append(copy(j))
        
    
    all_db[c_id] = db[c_id]