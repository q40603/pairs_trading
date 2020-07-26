#!/usr/bin/python3

from requests import get
from bs4 import BeautifulSoup as bs
from urllib.parse import unquote
from sklearn.feature_extraction.text import TfidfVectorizer
import cgi, json, datetime, ckiptagger, pickle, lzma, sys, codecs

field = cgi.FieldStorage()
content, models, vocabs, corpus = [], {}, {}, {}
code_to_name = dict(i.decode('utf-8').strip().split(',') for i in open('stock_codec.csv', 'rb'))
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
try:
    ws = ckiptagger.WS('./data', True)
except Exception as e:
    ckiptagger.data_utils.download_data_url('./')
    ws = ckiptagger.WS('./data', True)

try:
    src, stock = field.getvalue('src').split(','), field.getvalue('stock').split(',')
    #src, stock = ['ctee'], ['2330']
except:
    print('content-type:text/plain;charset:utf-8\n\nNo newspaper source or stock specified.')
    exit()

try:
    for st in stock:
        models[st] = pickle.loads(lzma.open(f'lr_model/{st}_model.pkl.xz', 'rb').read())
        vocabs[st] = pickle.loads(lzma.open(f'vocab/{st}_word.pkl.xz', 'rb').read())
        corpus[st] = [i.decode() for i in lzma.open(f'news/{st}.txt.xz', 'rb').readlines()]
except Exception as e:
    print('content-type:text/plain;charset:utf-8\n\nThere is no model for the specified stock(s).')
    print(e)
    exit()

def yahoo(content, stock):
    b = bs(get('https://tw.stock.yahoo.com/q/h?s=' + stock).text, 'lxml')
    for i in b.find('table',{'width':'100%'}).findAll('tr')[::2]:
        url = unquote('https://tw.stock.yahoo.com/' + i.find('a')['href'])
        title = i.text.split('\n')[2]
        B = bs(get(url).text, 'lxml')
        time = B.find('time')['datetime']
        text = ' '.join(j.text for j in B.findAll('p'))
        content.append({'url': url, 'title': title, 'time': time, 'text': text, 'stock': stock})

def ctee(content, qstock):
    stock = code_to_name[qstock]
    b = bs(get('https://ctee.com.tw/tag/' + stock).text, 'lxml')
    for i in b.findAll('article'):
        title = i.find('h2').text.replace('\n', '').replace('\t', '')
        url = i.find('h2').find('a')['href']
        time = i.find('time')['datetime']
        excerpt = i.find('div', class_='post-summary').text.strip()
        B = bs(get(url).text, 'lxml')
        text = ' '.join(j.text for j in B.find('div', class_='entry-content clearfix single-post-content').findAll('p'))
        content.append({'url': url, 'title': title, 'time': time, 'text': text, 'stock': qstock})

def udn(content, qstock):
    stock = code_to_name[qstock]
    b = bs(get('https://money.udn.com/search/tagging/1001/' + stock).text, 'lxml')
    for i in b.find(id='search_content').findAll('a', {'target': '_blank'}):
        url = i['href']
        title = i.find('h3').text
        B = bs(get(url).text, 'lxml')
        time = B.find('div',class_='shareBar__info--author').find('span').text
        text = ' '.join(i.text.strip() for i in B.find(id='article_body').findAll('p'))
        content.append({'url': url, 'title': title, 'time': time, 'text': text, 'stock': qstock})

print('content-type:application/json\naccess-control-allow-origin:*\n')
#print('content-type:text/plain; charset=utf-8\naccess-control-allow-origin:*\n')
for st in stock:
    for sr in src:
        if sr == 'ctee':
            ctee(content, st)
        elif sr == 'yahoo':
            yahoo(content, st)
        elif sr == 'udn':
            udn(content, st)

try:
    for st in stock:
        query = [news for news in content if news['stock'] == st]
        vec = TfidfVectorizer(token_pattern=r'(?u)\b\S+\b', vocabulary=vocabs[st])
        tfidf = vec.fit_transform(corpus[st] + [q['text'] for q in query]).toarray()
        for q, t in zip(query, tfidf[-len(query):]):
            q['verdict'] = 'red_k' if models[st].predict([t]) == 1 else 'black_k'
            q['prob'] = models[st].predict_proba([t]).tolist()
            q['good_word'] = list(set(w for w in ws([q['text']])[0] if w in vocabs[st] and 
                    models[st].coef_[0][vocabs[st][w]] > 0.5))
            q['bad_word'] = list(set(w for w in ws([q['text']])[0] if w in vocabs[st] and 
                    models[st].coef_[0][vocabs[st][w]] < -0.5))
except Exception as e:
    print(e); exit()

print(json.dumps(content, ensure_ascii=False))
