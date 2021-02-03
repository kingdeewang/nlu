#! -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from margin_softmax import *
from keras.layers import *
from flask import Flask
import requests
import json
from bs4 import BeautifulSoup
from keras.models import Model
from vib import VIB
from keras.constraints import unit_norm
import os
import logging
from tqdm import tqdm
from urllib.parse import quote
from sys import version_info
import urllib
import random
import re
import json

logging.basicConfig(filename=os.path.join(os.getcwd(),'log/log.txt'),level=logging.INFO)

app = Flask(__name__)

model = None
encoder = None
chars={}
id2char={}
char2id={}
min_count = 2
maxlen=30
word_size = 128
num_train_groups =76# 前9万组问题拿来做训练

data = pd.read_csv('data/tongyiju.csv', encoding='utf-8', header=None, delimiter='\t')

def strQ2B(ustring): # 全角转半角
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288: # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


data[1] = data[1].apply(strQ2B)
data[1] = data[1].str.lower()
chars = {}
for s in tqdm(iter(data[1])):
    for c in s:
        if c not in chars:
            chars[c] = 0
        chars[c] += 1
chars = {i:j for i,j in chars.items() if j >= min_count}

def getRecords(url):
    session = requests.session()
    session.keep_alive = False
    headers={
        'Host': 'jandan.net',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
         'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
         'Accept-Encoding': 'gzip, deflate',
    }
    wb_data_sub = requests.get(url)
    soup = BeautifulSoup(wb_data_sub.text, 'lxml')
    data = soup.p.string
    text = json.loads(data)
    records =text['response']['docs']
    return records

def string2id(s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]
    _ = _ + [0] * (maxlen - len(_))
    return _



#定义文本相似度
def most_similar(ss):
     global encoder
     p=re.compile(r'[$()#+&*]')
     ss = re.sub(p,"",ss)
     ss = ss.replace(u"？","")
     tt = quote(ss.encode('utf-8'))
     #logging.info(tt)
     url = "http://localhost:8983/solr/semantic/select?q="+tt+"&wt=json"
     logging.info(url)
     datas = getRecords(url)
     if(len(datas) == 0):
         return ""

     keywords_vec =[]
     dict={}
     answer = []
     for data in datas:
         if not 'keywords' in data:
             continue

         tt = data['keywords'].split(";")
         for keyword in tt:
            dict={}
            dict['keywords'] = keyword
            if 'content' in data:
                 content = data['content']
                 if type(content).__name__ == 'list':
                     content = content[0]
                 dict['answer'] = content
            else:
                 dict['answer'] =''

            if 'category' in data:
                dict['category'] = data['category']
            else:
                continue

         keywords_vec.append(string2id(keyword))
         answer.append(dict)


     if encoder is None:
        x_in = Input(shape=(maxlen,))
        x_embedded = Embedding(len(chars)+2,
                       word_size)(x_in)
        x =  Bidirectional(GRU(word_size), merge_mode = 'ave')(x_embedded)
        x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        encoder = Model(x_in, x) # 最终的目的是要得到一个编码器
        encoder.load_weights("gru_data/sim_encoder_weight.h5")

     valid_vec = encoder.predict(np.array(list(keywords_vec)),
                            verbose=True,
                            batch_size=100) # encoder计算句向量
     v = encoder.predict(np.array([string2id(ss)]))[0]
     sims = np.dot(valid_vec, v)
     #for i in range(len(sims)):
     #    logging.info("%s,%s", answer[i]['keywords'],sims[i])

     for i in range(len(sims)):
         if sims[i] > 0.91 :
             dict = answer[i]
             print (dict)
             return json.dumps(dict)
     return ''

@app.route('/getAnswer/<sentence>')
def show_sentence_domain1(sentence):
     content = most_similar(sentence)
     return content

@app.route('/domain/<sentence>')
def show_sentence_domain(sentence):
    global model
    global char2id
    global encoder
    global label

    if(model is None):
        if(len(char2id) == 0):
                x_train,y_train,char2id,combined,label = pickle.load(open('gru_data/data.config', 'rb'))

        #print(model.to_json())  #秒级时间戳
            # 正式模型，基于GRU的分类器
        x_in = Input(shape=(maxlen,))
        x_embedded = Embedding(len(char2id)+2,
                               word_size, mask_zero=True)(x_in)
        x = Bidirectional(GRU(word_size))(x_embedded)
        # x = Bidirectional(GRU(word_size))(x)
        #x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
        #x= AttentionLayer()(x)
        #encoder = Model(x_in, x) # 最终的目的是要得到一个编码器
        z_mean = Dense(128)(x)
        z_log_var = Dense(128)(x)
        x = VIB(0.1)([z_mean, z_log_var])
        x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        pred = Dense(len(label),
                     use_bias=False,
                     kernel_constraint=unit_norm())(x)

        model = Model(x_in, pred) # 用分类问题做训练
        model.load_weights("gru_data/classfication_weight.h5")

    x_predict = np.array([string2id(sentence)])
    predict = model.predict(x_predict)



    logging.info("%s,%s %f" %(sentence,label[np.argmax(predict)],predict[0][np.argmax(predict)]))
    threshold = 0.55
    if(label[np.argmax(predict)] == 'recall' or label[np.argmax(predict)] == 'error'):
        if len(sentence) < 15:
            threshold = 0.7
        else:
            threshold = 0.6

    if(predict[0][np.argmax(predict)]  > threshold):
        return label[np.argmax(predict)]
    return ""

@app.route("/index")
def index():
    return "idex"

if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run()
