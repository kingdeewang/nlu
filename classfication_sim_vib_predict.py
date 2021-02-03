#! -*- coding: utf-8 -*-
import os
import pickle
from keras.models import Model
from margin_softmax import *
from keras.models import load_model
from tqdm import tqdm
from keras.layers import *
from keras.constraints import unit_norm
import json
import requests
from bs4 import BeautifulSoup
import urllib3
import urllib
#import urllib.request
from vib import VIB
from sys import version_info

if version_info == 2:
    chr = unichr
else:
    chr = chr


chars={}
id2char={}
num_train_groups =74# 前9万组问题拿来做训练
maxlen = 30
batch_size = 150
min_count = 10
word_size = 128
epochs = 30 # amsoftmax需要25个epoch，其它需要20个epoch
solrUrl = "http://localhost:8983/solr/semantic/select?q="

def string2id(s):
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
        rstring += chr(inside_code)
    return rstring

def getRecords(url):
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
     url = solrUrl +ss + "&wt=json"
     print(url)
     datas =getRecords(url)

     if(len(datas) == 0):
         return ""

     keywords_vec =[]
     keywords = []
     contents =[]
     for data in datas:
         tt = data['keywords'].split(";")
         for keyword in tt:
            keywords.append(keyword)
            keywords_vec.append(string2id(keyword))
            if 'content' in data:
                 content = str(data['content'])
                 content=content.split(";")[0]
                 contents.append(content)
            else:
                 contents.append("")
     encoder =  load_model("gru_data/sim.h5",custom_objects={'sparse_amsoftmax_loss':sparse_amsoftmax_loss})
     valid_vec = encoder.predict(np.array(list(keywords_vec)),
                            verbose=True,
                            batch_size=32) # encoder计算句向量

     v = encoder.predict(np.array([string2id(ss)]))[0]
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


     def simstring2id(s):
        _ = [char2id.get(i, 1) for i in s[:maxlen]]
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

if os.path.exists('gru_data/data.config'):  # 如果有读取数据文件
    x_train,y_train,char2id,combined = pickle.load(open('gru_data/data.config', 'rb'))
else:
    print("error to input parameters")

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
pred = Dense(num_train_groups,
             use_bias=False,
             kernel_constraint=unit_norm())(x)

model = Model(x_in, pred) # 用分类问题做训练
model.load_weights("gru_data/classfication_weight.h5")
aa =[u"你知道我叫什么名字吗",u"播放电影保镖的主题曲",u"爱上英语课中国你怕警察",u"而且回来也不干",u"杭州明天温度怎么样",u"男人肾虚怎么治？",u"中山有鸡巴是受了啥大家继续爱他多喝点多少岁参加的推论"]


for ss in aa:
    x_predict = np.array([string2id(ss)])
    predict = model.predict(x_predict)
    label =['audio', 'music','genre', 'cmds','poi','chat','stock','localsearch','traffic','weather',
            'sms','wechat','call','joke','time','date','train','volume','cal' ,'news',
            'illegal','bc','greet','oilprice','translate','location','name_recognize','qa','recommend','exchage',
            'traffic_controler','navi',
            #'sex','age','name'
            'localsearch_food','confirm','no'
            ,'poem','start_navi','quit_navi','play_info','play_stop',
            'play_random','billboard','route','distance','costtime','website','websearch','recall','error','cancel',
            'uncertain','quit_app','movie','appmgr','play_switch','power_info','position','qq_music',
            'entertain','play_local','video','alarm','flight','sports','tv','tag',
            'call_redial','lottery','resume','play_mode','red_music','switch_screen','play_card','name_rember']
    # label =['audio', 'music','genre', 'cmds','poi','navi','chat','stock','localsearch','traffic','weather','sms','wechat','joke','times','date','train','call','volume',
    #         'website','cal' ,'qa','greet']
    print(predict[0])
    if(predict[0][np.argmax(predict)] < 0.5):
        print("%s,%s" %(ss,label[5]))
    else:
        print("%s,%s" %(ss,label[np.argmax(predict)]))

 #   if label[np.argmax(predict)] =='chat' or label[np.argmax(predict)] =='qa' or label[np.argmax(predict)] =='name':
 #       question = most_similar(ss)
 #       print(question)
