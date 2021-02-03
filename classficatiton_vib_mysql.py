# ! -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import pandas as pd
import pickle
import json
from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from margin_softmax import *
from keras.models import load_model
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from vib import VIB
from sys import version_info
import random
import pymysql


chars = {}
id2char = {}
maxlen = 30
batch_size = 150
min_count = 10
word_size = 128
print("test")
dbconn = pymysql.connect(
  host="localhost",
  database="corpus",
  user="root",
  password="1",
  port=3306,
  charset='utf8'
)

label2Index = {}
category = []


def loadData():
    combined = []
    y_train = []
    sqlcmd = "select sentence, context, service from tbl_service order by service"
    df = pd.read_sql(sqlcmd, dbconn)
    rs = df.values
    for data in rs:
        sentence = data[0]
        if data[1] != '':
            sentence = sentence + "|" + data[1];
        combined.append(sentence)
        if not category.__contains__(data[2]):
            category.append(data[2])
            label2Index = {key : i for i, key in enumerate(category)}

        y_train.append(label2Index[data[2]])

    for s in combined:
        for c in s:
            if c not in chars:
                chars[c] = 0
            chars[c] += 1

    # 0: padding标记
    # 1: unk标记
    charslist = {i:j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2:j for i, j in enumerate(charslist)}
    char2id = {j:i for i, j in id2char.items()}
    x_train = np.array([string2id(char2id, s) for s in combined])
    pickle.dump([x_train, y_train, char2id, combined, category], open('gru_data/data.config', 'wb'))

    return x_train, y_train, char2id, combined


def strQ2B(ustring):  # 全角转半角
    rstring = ''
    for uchar in ustring:
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def string2id(char2id, s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]
    _ = _ + [0] * (maxlen - len(_))
    return _

# 正式模型，基于双向GRU的分类器
x_train, y_train, char2id, combined = loadData()

x_in = Input(shape=(maxlen,))
x_embedded = Embedding(len(char2id) + 2,
                       word_size, mask_zero=True)(x_in)
x = Bidirectional(GRU(word_size))(x_embedded)
z_mean = Dense(128)(x)
z_log_var = Dense(128)(x)
x = VIB(0.1)([z_mean, z_log_var])
x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
pred = Dense(len(category),
             use_bias=False,
             kernel_constraint=unit_norm())(x)

model = Model(x_in, pred)  # 用分类问题做训练

def train_cnn(x_train, y_train, char2id, epochs):

    model.compile(loss=sparse_amsoftmax_loss,
                   optimizer='adam',
                   metrics=['sparse_categorical_accuracy'])

    # 定义Callback器，计算验证集的acc，并保存最优模型
    class Evaluate(Callback):

        def __init__(self):
            self.accs = {'top1': [], 'top5': [], 'top10': []}
            self.highest = 0.

        def on_epoch_end(self, epoch, logs=None):
            top1_acc, top5_acc, top10_acc = evaluate()
            self.accs['top1'].append(top1_acc)
            self.accs['top5'].append(top5_acc)
            self.accs['top10'].append(top10_acc)
            if top1_acc >= self.highest:  # 保存最优模型权重
                self.highest = top1_acc
                model.save_weights('sent_sim_amsoftmax.model')
            json.dump({'accs': self.accs, 'highest_top1': self.highest},
                      open('valid_amsoftmax.log', 'w'), indent=4)
            print ('top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc))

    evaluator = Evaluate()
    #randnum = random.randint(0,100)
    #random.seed(randnum)
    #random.shuffle(x_train)
    x_test = x_train[-500:]
    #random.seed(randnum)
    #random.shuffle(y_train)
    y_test = y_train[-500:]
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        # validation_split=0.05
                        )

    model.save_weights("gru_data/classfication_weight.h5")
    # model.predict(u"还有多少电")

def training(epochs):

    # id与组别之间的映射
    train_cnn(x_train, y_train, char2id, epochs)

if __name__ == '__main__':
    training(25)
    # if os.path.exists('gru_data/data.config'):  # 如果有读取数据文件
    #     x_train,y_train,char2id,combined = pickle.load(open('gru_data/data.config', 'rb'))
    # else:
else:
    model.load_weights("gru_data/classfication_weight.h5")


