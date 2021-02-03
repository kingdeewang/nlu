#! -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from margin_softmax import *
from keras.models import load_model
from tqdm import tqdm
from keras.layers import *
import requests
import urllib.request
import json
from bs4 import BeautifulSoup
import urllib3

chars={}
id2char={}
min_count = 2
maxlen=30
word_size = 128
solrUrl = "http://localhost:8983/solr/semantic/select?q="

def string2id(char2id,s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]
    _ = _ + [0] * (maxlen - len(_))
    return _

def strQ2B(ustring): # 全角转半角
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288: # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring

def getRecords(url):
    wb_data_sub = requests.get(url)
    soup = BeautifulSoup(wb_data_sub.text, 'lxml')
    data = soup.p.string
    text = json.loads(data)
    records =text['response']['docs']
    return records

#定义文本相似度
def most_similar(ss):
     url = solrUrl + ss + "&wt=json"
     print(url)
     data =getRecords(url)

     if(len(data) == 0):
         return ""

     chars = {}
     for s in tqdm(iter(data)):
        for c in s['keywords']:
            if c not in chars:
                chars[c] = 0
        chars[c] += 1

     # 0: padding标记
     # 1: unk标记
     simchars = {i:j for i,j in chars.items() if j >= min_count}
     simid2char = {i+2:j for i,j in enumerate(simchars)}
     simchar2id = {j:i for i,j in simid2char.items()}

     def simstring2id(s):
        _ = [simchar2id.get(i, 1) for i in s[:maxlen]]
        _ = _ + [0] * (maxlen - len(_))
        return _

     keywords_vec = [simstring2id(ss['keywords']) for ss in data]
     keywords = [ss['keywords'] for ss in data]
     contents = [ss['content'] for ss in data]

     encoder =  load_model("gru_data/sim.h5",custom_objects={'sparse_amsoftmax_loss':sparse_amsoftmax_loss})
     if os.path.exists('gru_data/valid_vec.config'):  # 如果有读取数据文件
         valid_vec = pickle.load(open('gru_data/valid_vec.config', 'rb'))
     else:
         valid_vec = encoder.predict(np.array(list(keywords_vec)),
                            verbose=True,
                            batch_size=100) # encoder计算句向量
         pickle.dump(valid_vec, open('gru_data/valid_vec.config', 'wb'))

     v = encoder.predict(np.array([simstring2id(ss)]))[0]
     sims = np.dot(valid_vec, v)
     for i in sims.argsort()[-5:][::-1]:
        return keywords[i],contents[i],sims[i]


def getAnswer(ss):
     if os.path.exists('gru_data/qa.config'):  # 如果有读取数据文件
         data = pickle.load(open('gru_data/qa.config', 'rb'))
     else:
         from os import listdir
         filename_list=listdir("qa_data")
         data=[]

         #可以同过简单后缀名判断，筛选出你所需要的文件
         for filename in filename_list:#依次读入列表中的内容
             if filename.endswith('.json'):# 后缀名'jpg'匹对
                 #print(filename)
                 with open('qa_data/'+filename,'r') as json_file:
                     temp = json.loads(json_file.read())
                     data.extend(temp)
         pickle.dump(data, open('gru_data/qa.config', 'wb'))
     chars = {}
     question=[]
     answer=[]
     for qa in tqdm(iter(data)):
        question.append(qa['question'])
        answer.append(qa['answer'])
        for c in qa['question']:
            if c not in chars:
                chars[c] = 0
        chars[c] += 1

     simchars = {i:j for i,j in chars.items() if j >= min_count}
     simid2char = {i+2:j for i,j in enumerate(simchars)}
     simchar2id = {j:i for i,j in simid2char.items()}

     def simstring2id(s):
        _ = [simchar2id.get(i, 1) for i in s[:maxlen]]
        _ = _ + [0] * (maxlen - len(_))
        return _

     #ss = ss.replace("，","")
     valid_data = [simstring2id(s) for s in question]
     encoder =  load_model("gru_data/sim.h5",custom_objects={'sparse_amsoftmax_loss':sparse_amsoftmax_loss})
     valid_vec = encoder.predict(np.array(valid_data),
                            verbose=True,
                            batch_size=100) # encoder计算句向量
     v = encoder.predict(np.array([simstring2id(ss)]))[0]
     sims = np.dot(valid_vec, v)
     for i in sims.argsort()[-5:][::-1]:
         print (question[i],answer[i],sims[i])


model = load_model("gru_data/classfication.h5",custom_objects={'sparse_amsoftmax_loss':sparse_amsoftmax_loss})
aa =[u"退出吧东东",u"帮我三点",u'call;帮我回家',u'这是什么歌',u"到连云港怎么走",u"去合肥多长时间",u"请帮我找个美女图片"]

if os.path.exists('gru_data/data.config'):  # 如果有读取数据文件
    x_train,y_train,char2id,combined = pickle.load(open('gru_data/data.config', 'rb'))

for ss in aa:
    x_predict = np.array([string2id(char2id,ss)])
    predict = model.predict(x_predict)
    print(predict)
    label =['audio', 'music','genre', 'cmds','poi','chat','stock','localsearch','traffic','weather','sms','wechat','call','joke','time','date','train','volume','cal'
        ,'news','illegal','bc','greet','oilprice','translate','location','name','qa','recommend','exchage','traffic_controler','navi',
            'sex','age','name','poem','start_navi','quit_navi','play_info','play_stop','play_random','billboard','route','distance','costtime','website',
            'websearch','recall','error','cancel','uncertain','quit_app','audio','movie','appmgr']
    # label =['audio', 'music','genre', 'cmds','poi','navi','chat','stock','localsearch','traffic','weather','sms','wechat','joke','times','date','train','call','volume',
    #         'website','cal' ,'qa','greet']
    if(predict[0][np.argmax(predict)] < 0.45):
        print("%s,%s" %(ss,label[5]))
    else:
        print("%s,%s" %(ss,label[np.argmax(predict)]))

    #if label[np.argmax(predict)] =='chat' or label[np.argmax(predict)] =='qa' or label[np.argmax(predict)] =='name':
       #question = most_similar(ss)
     #  print(question)
