#! -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
import json
from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from margin_softmax import *
from keras.callbacks import Callback
from vib import VIB
num_train_groups =36000 # 前9万组问题拿来做训练
word_size = 128
min_count =2
maxlen = 30

data = pd.read_csv('data/tongyiju.csv', encoding='utf-8', header=None, delimiter='\t')


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


data[1] = data[1].apply(strQ2B)
data[1] = data[1].str.lower()

chars = {}
for s in tqdm(iter(data[1])):
    for c in s:
        if c not in chars:
            chars[c] = 0
        chars[c] += 1


# 0: padding标记
# 1: unk标记
chars = {i:j for i,j in chars.items() if j >= min_count}
id2char = {i+2:j for i,j in enumerate(chars)}
char2id = {j:i for i,j in id2char.items()}

def string2id(s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]
    _ = _ + [0] * (maxlen - len(_))
    return _

data[2] = data[1].apply(string2id)
valid_data = data

# 正式模型，基于GRU的分类器
x_in = Input(shape=(maxlen,))
x_embedded = Embedding(len(chars)+2,
                       word_size)(x_in)
x = Bidirectional(GRU(word_size), merge_mode = 'ave')(x_embedded)
x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
z_mean = Dense(128)(x)
z_log_var = Dense(128)(x)
#x = VIB(0.1)([z_mean, z_log_var])

#model = Model(x_in, pred) # 用分类问题做训练
#model.load_weights("sent_sim_amsoftmax.model")
encoder = Model(x_in, x) # 最终的目的是要得到一个编码器
#encoder = Model(x_in, x) # 最终的目的是要得到一个编码器
encoder.load_weights("gru_data/sim_encoder_weight.h5")
valid_vec = encoder.predict(np.array(list(valid_data[2])),
                            verbose=True,
                            batch_size=100) # encoder计算句向量

def most_similar(s):
    v = encoder.predict(np.array([string2id(s)]))[0]
    sims = np.dot(valid_vec, v)
    for i in sims.argsort()[-5:][::-1]:
        print (valid_data.iloc[i][1],sims[i])


most_similar(u'现在几点了')
most_similar(u'你妈妈是谁')
most_similar(u'你爷爷是谁')