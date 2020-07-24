from flask import Flask, render_template, jsonify, request
from myTools.pairstrading.pairs_trading import *
import json  

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("pairs_trading.html")


@app.route("/stock/find_past_pairs",methods=['GET'])
def find_past_pairs():
    trade_date = request.values.get('trade_date')
    data = get_all_pairs(trade_date)
    print(data)
    return jsonify(data)

@app.route("/stock/get_pairs_price",methods=['GET'])
def get_pairs_price():
    s1 = request.values.get('s1')
    s2 = request.values.get('s2')
    trade_date = request.values.get('trade_date')
    w1 = request.values.get('w1')
    w2 = request.values.get('w2')
    data = get_pairs_spread(trade_date, s1, s2, float(w1), float(w2))
    tmp = get_s_name(s1.replace("s_",""),s2.replace("s_",""))
    data["s1_info"] = tmp[0]
    data["s2_info"] = tmp[1]
    data = json.dumps(data,default= str)
    return jsonify(data)

@app.route("/stock/trade_backtest",methods=['GET'])  
def trade_backtest():
    s1 = request.values.get('s1')
    s2 = request.values.get('s2')
    choose_date = request.values.get('trade_date')
    capital = 3000           # 每組配對資金300萬
    maxi = 5                 # 股票最大持有張數
    open_time = 1.5                 # 開倉門檻倍數
    stop_loss_time = 2.5                  # 停損門檻倍數
    tax_cost = 0
    pair_list = [[s1, s2]]
    data = trade_certain_pairs(choose_date, capital, maxi, open_time, stop_loss_time, tax_cost, pair_list)
    data["s1_news"] = get_stock_news(choose_date,s1)
    data["s2_news"] = get_stock_news(choose_date,s2)
    data = json.dumps(data,default= str)
    return jsonify(data)
    
    
if __name__ == "__main__":
    app.run(debug=True)