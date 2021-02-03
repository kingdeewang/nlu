#! -*- coding: utf-8 -*-
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

def string2id(char2id,s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]
    _ = _ + [0] * (maxlen - len(_))
    return _

def loadfile():
    audio = pd.read_csv('data/audio.txt',  encoding='utf-8', header=None)
    music = pd.read_csv('data/music.txt', encoding='utf-8', header=None)
    genre = pd.read_csv('data/genre.txt', encoding='utf-8', header=None)
    cmds = pd.read_csv('data/cmd.txt',  encoding='utf-8', header=None)
    poi = pd.read_csv('data/poi.txt',  encoding='utf-8', header=None)
    chat = pd.read_csv('data/chat.txt',  encoding='utf-8', header=None)
    stock = pd.read_csv('data/stock.txt',  encoding='utf-8', header=None)
    localsearch = pd.read_csv('data/localsearch.txt',  encoding='utf-8', header=None)
    traffic = pd.read_csv('data/traffic.txt',  encoding='utf-8', header=None)
    weather = pd.read_csv('data/weather.txt',  encoding='utf-8', header=None)
    sms = pd.read_csv('data/sms.txt',  encoding='utf-8', header=None)
    wechat = pd.read_csv('data/wechat.txt',  encoding='utf-8', header=None)
    joke = pd.read_csv('data/joke.txt',  encoding='utf-8', header=None)
    times = pd.read_csv('data/time.txt',  encoding='utf-8', header=None)
    date = pd.read_csv('data/date.txt',  encoding='utf-8', header=None)
    train = pd.read_csv('data/train.txt', encoding='utf-8', header=None)
    call = pd.read_csv('data/call.txt', encoding='utf-8', header=None)
    volume = pd.read_csv('data/volume.txt', encoding='utf-8', header=None)
    cal = pd.read_csv('data/cal.txt', encoding='utf-8', header=None)
    news = pd.read_csv('data/news.txt', encoding='utf-8', header=None)
    illegal = pd.read_csv('data/illegal.txt', encoding='utf-8', header=None)
    bc = pd.read_csv('data/bc.txt', encoding='utf-8', header=None)
    greet = pd.read_csv('data/greet.txt', encoding='utf-8', header=None)
    oilprice = pd.read_csv('data/oilprice.txt', encoding='utf-8', header=None)
    translate = pd.read_csv('data/translate.txt', encoding='utf-8', header=None)
    location = pd.read_csv('data/location.txt', encoding='utf-8', header=None)
    name_recognize = pd.read_csv('data/name_recognize.txt', encoding='utf-8', header=None)
    name_rember = pd.read_csv('data/name_rember.txt', encoding='utf-8', header=None)
    qa = pd.read_csv('data/qa.txt', encoding='utf-8', header=None)
    recommend = pd.read_csv('data/recommend.txt', encoding='utf-8', header=None)
    exchange = pd.read_csv('data/exchange.txt', encoding='utf-8', header=None)
    traffic_control = pd.read_csv('data/traffic_control.txt', encoding='utf-8', header=None)
    navi = pd.read_csv('data/navi.txt', encoding='utf-8', header=None)
    #sex = pd.read_csv('data/sex.txt', encoding='utf-8', header=None)
    #age = pd.read_csv('data/age.txt', encoding='utf-8', header=None)
    #name = pd.read_csv('data/name.txt', encoding='utf-8', header=None)
    poem = pd.read_csv('data/poem.txt', encoding='utf-8', header=None)
    start_navi  = pd.read_csv('data/start_navi.txt', encoding='utf-8', header=None)
    quit_navi  = pd.read_csv('data/quit_navi.txt', encoding='utf-8', header=None)
    play_info  = pd.read_csv('data/play_info.txt', encoding='utf-8', header=None)
    play_stop  = pd.read_csv('data/play_stop.txt', encoding='utf-8', header=None)
    play_random = pd.read_csv('data/play_random.txt', encoding='utf-8', header=None)
    billboard  = pd.read_csv('data/billboard.txt', encoding='utf-8', header=None)
    route  = pd.read_csv('data/route.txt', encoding='utf-8', header=None)
    distance  = pd.read_csv('data/distance.txt', encoding='utf-8', header=None)
    costtime  = pd.read_csv('data/costtime.txt', encoding='utf-8', header=None)
    website  = pd.read_csv('data/website.txt', encoding='utf-8', header=None)
    websearch  = pd.read_csv('data/websearch.txt', encoding='utf-8', header=None)
    recall = pd.read_csv('data/recall.txt', encoding='utf-8', header=None)
    movie = pd.read_csv('data/movie.txt', encoding='utf-8', header=None)
    error = pd.read_csv('data/error.txt', encoding='utf-8', header=None)
    cancel = pd.read_csv('data/cancel.txt', encoding='utf-8', header=None)
    uncertain = pd.read_csv('data/uncertain.txt', encoding='utf-8', header=None)
    quit_app = pd.read_csv('data/quit_app.txt', encoding='utf-8', header=None)
    appmgr = pd.read_csv('data/appmgr.txt', encoding='utf-8', header=None)
    play_switch = pd.read_csv('data/play_switch.txt', encoding='utf-8', header=None)
    power_info = pd.read_csv('data/power.txt', encoding='utf-8', header=None)
    position = pd.read_csv('data/position.txt', encoding='utf-8', header=None)
    localsearch_food = pd.read_csv('data/localsearch_food.txt', encoding='utf-8', header=None)
    confirm = pd.read_csv('data/confirm.txt', encoding='utf-8', header=None)
    qq_music = pd.read_csv('data/qq_music.txt', encoding='utf-8', header=None)
    no = pd.read_csv('data/no.txt', encoding='utf-8', header=None)
    entertain = pd.read_csv('data/entertain.txt', encoding='utf-8', header=None)
    play_local = pd.read_csv('data/play_local.txt', encoding='utf-8', header=None)
    video = pd.read_csv('data/video.txt', encoding='utf-8', header=None)
    alarm = pd.read_csv('data/alarm.txt', encoding='utf-8', header=None)
    flight = pd.read_csv('data/flight.txt', encoding='utf-8', header=None)
    sports = pd.read_csv('data/sports.txt', encoding='utf-8', header=None)
    tv = pd.read_csv('data/tv.txt', encoding='utf-8', header=None)
    tag = pd.read_csv('data/tag.txt', encoding='utf-8', header=None)
    lottery = pd.read_csv('data/lottery.txt', encoding='utf-8', header=None)
    resume = pd.read_csv('data/resume.txt', encoding='utf-8', header=None)
    call_redial = pd.read_csv('data/call_redial.txt', encoding='utf-8', header=None)
    play_mode = pd.read_csv('data/play_mode.txt', encoding='utf-8', header=None)
    red_music = pd.read_csv('data/red_music.txt', encoding='utf-8', header=None)
    switch_screen = pd.read_csv('data/switch_screen.txt', encoding='utf-8', header=None)
    play_card = pd.read_csv('data/play_card.txt', encoding='utf-8', header=None)
    combined = np.concatenate((music[0],genre[0], cmds[0],poi[0],chat[0],stock[0],localsearch[0],traffic[0],weather[0],
                               sms[0], wechat[0], call[0],
                               joke[0],times[0],date[0],train[0],volume[0],cal[0],news[0],illegal[0],bc[0],greet[0],oilprice[0],
                               translate[0],location[0],name_recognize[0],qa[0],recommend[0],exchange[0],traffic_control[0],navi[0],
                               #sex[0],age[0],name[0],
                               poem[0],start_navi[0],quit_navi[0],play_info[0],play_stop[0],
                               play_random[0],billboard[0],route[0],distance[0],costtime[0],website[0],websearch[0],recall[0],error[0],cancel[0],
                               uncertain[0],quit_app[0],movie[0],appmgr[0],play_switch[0],power_info[0],position[0],localsearch_food[0],confirm[0],
                               no[0],qq_music[0],entertain[0],play_local[0],video[0],alarm[0],flight[0],sports[0],tv[0],tag[0],call_redial[0],lottery[0],resume[0],
                               play_mode[0],red_music[0],switch_screen[0],play_card[0],name_rember[0],audio[0]))


    print(combined.shape)

    for s in combined:
        for c in s:
            if c not in chars:
                chars[c] = 0
            chars[c] += 1

    # 0: padding标记
    # 1: unk标记
    charslist = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(charslist)}
    char2id = {j:i for i,j in id2char.items()}

    x_train =np.array([string2id(char2id,s) for s in combined])
    audio_array = np.array([0]*len(audio),dtype=int)
    music_array = np.array([1]*len(music),dtype=int)
    genre_array = np.array([2]*len(genre),dtype=int)
    cmds_array = np.array([3]*len(cmds),dtype=int)
    poi_array = np.array([4]*len(poi),dtype=int)
    chat_array = np.array([5]*len(chat),dtype=int)
    stock_array = np.array([6]*len(stock),dtype=int)
    localsearch_array = np.array([7]*len(localsearch),dtype=int)
    traffic_array = np.array([8]*len(traffic),dtype=int)
    weather_array = np.array([9]*len(weather),dtype=int)
    sms_array = np.array([10]*len(sms),dtype=int)
    wechat_array = np.array([11]*len(wechat),dtype=int)
    call_array = np.array([12]*len(call),dtype=int)
    joke_array = np.array([13]*len(joke),dtype=int)
    time_array = np.array([14]*len(times),dtype=int)
    date_array = np.array([15]*len(date),dtype=int)
    train_array = np.array([16]*len(train),dtype=int)
    volume_array = np.array([17]*len(volume),dtype=int)
    cal_array = np.array([18]*len(cal),dtype=int)
    news_array = np.array([19]*len(news),dtype=int)
    illegal_array = np.array([20]*len(illegal),dtype=int)
    bc_array = np.array([21]*len(bc),dtype=int)
    greet_array = np.array([22]*len(greet),dtype=int)
    oilprice_array = np.array([23]*len(oilprice),dtype=int)
    translate_array = np.array([24]*len(translate),dtype=int)
    location_array = np.array([25]*len(location),dtype=int)
    name_recognize_array = np.array([26]*len(name_recognize),dtype=int)
    qa_array = np.array([27]*len(qa),dtype=int)
    recommend_array = np.array([28]*len(recommend),dtype=int)
    exchange_array = np.array([29]*len(exchange),dtype=int)
    traffic_control_array = np.array([30]*len(traffic_control),dtype=int)
    navi_array = np.array([31]*len(navi),dtype=int)
    #sex_array = np.array([32]*len(sex),dtype=int)
    #age_array = np.array([33]*len(age),dtype=int)
    #name_array = np.array([34]*len(name),dtype=int)
    localsearch_food_array = np.array([32]*len(localsearch_food),dtype=int)
    confirm_array = np.array([33]*len(confirm),dtype=int)
    no_array = np.array([34]*len(no),dtype=int)
    poem_array = np.array([35]*len(poem),dtype=int)
    start_navi_array = np.array([36]*len(start_navi),dtype=int)
    quit_navi_array = np.array([37]*len(quit_navi),dtype=int)
    play_info_array = np.array([38]*len(play_info),dtype=int)
    play_stop_array = np.array([39]*len(play_stop),dtype=int)
    play_random_array = np.array([40]*len(play_random),dtype=int)
    billboard_array = np.array([41]*len(billboard),dtype=int)
    route_array = np.array([42]*len(route),dtype=int)
    distance_array = np.array([43]*len(distance),dtype=int)
    costtime_array = np.array([44]*len(costtime),dtype=int)
    website_array = np.array([45]*len(website),dtype=int)
    websearch_array = np.array([46]*len(websearch),dtype=int)
    recall_array = np.array([47]*len(recall),dtype=int)
    error_array = np.array([48]*len(error),dtype=int)
    cancel_array = np.array([49]*len(cancel),dtype=int)
    uncertain_array = np.array([50]*len(uncertain),dtype=int)
    quit_app_array = np.array([51]*len(quit_app),dtype=int)
    movie_array = np.array([52]*len(movie),dtype=int)
    appmgr_array = np.array([53]*len(appmgr),dtype=int)
    playswitch_array = np.array([54]*len(play_switch),dtype=int)
    powerinfo_array = np.array([55]*len(power_info),dtype=int)
    position_array = np.array([56]*len(position),dtype=int)
    qq_music_array = np.array([57]*len(qq_music),dtype=int)
    entertain_array = np.array([58]*len(entertain),dtype=int)
    playlocal_array = np.array([59]*len(play_local),dtype=int)
    video_array = np.array([60]*len(video),dtype=int)
    alarm_array = np.array([61]*len(alarm),dtype=int)
    flight_array = np.array([62]*len(flight),dtype=int)
    sports_array = np.array([63]*len(sports),dtype=int)
    tv_array = np.array([64]*len(tv),dtype=int)
    tag_array = np.array([65]*len(tag),dtype=int)
    call_redial_array = np.array([66]*len(call_redial),dtype=int)
    lottery_array = np.array([67]*len(lottery),dtype=int)
    resume_array = np.array([68]*len(resume),dtype=int)
    play_mode_array = np.array([69]*len(play_mode),dtype=int)
    red_music_array = np.array([70]*len(red_music),dtype=int)
    switch_screen_array = np.array([71]*len(switch_screen),dtype=int)
    play_card_array = np.array([72]*len(play_card),dtype=int)
    name_rember_array = np.array([73]*len(name_rember),dtype=int)
    y_train = np.hstack((music_array,genre_array,cmds_array,poi_array,chat_array,stock_array,localsearch_array,traffic_array,weather_array,
                         sms_array,wechat_array, call_array,
                         joke_array,time_array,date_array,train_array,
                         volume_array,cal_array, news_array,illegal_array,bc_array,greet_array,
                         oilprice_array,translate_array, location_array,name_recognize_array,qa_array,
                         recommend_array,exchange_array,traffic_control_array,navi_array,
                         #sex_array,age_array,name_array,
                         poem_array,start_navi_array,quit_navi_array,play_info_array,
                         play_stop_array,play_random_array,billboard_array,
                         route_array,distance_array,costtime_array,website_array,websearch_array,recall_array,
                         error_array,cancel_array,uncertain_array,quit_app_array,movie_array,appmgr_array,
                         playswitch_array,powerinfo_array,position_array,localsearch_food_array,confirm_array,no_array,
                         qq_music_array,entertain_array,playlocal_array,video_array,alarm_array,flight_array,
                         sports_array,tv_array,tag_array,call_redial_array,lottery_array,resume_array,play_mode_array,
                         red_music_array,switch_screen_array,play_card_array,name_rember_array,audio_array))
    pickle.dump([x_train,y_train,char2id,combined], open('gru_data/data.config', 'wb'))
    return x_train,y_train,char2id,combined


def train_cnn(x_train,y_train,char2id):
    # 正式模型，基于GRU的分类器
    x_in = Input(shape=(maxlen,))
    x_embedded = Embedding(len(char2id)+2,
                           word_size, mask_zero=True)(x_in)
    x = Bidirectional(GRU(word_size))(x_embedded)
    # x = Bidirectional(GRU(word_size))(x)
    #x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
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
    import keras.optimizers
    adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
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
            if top1_acc >= self.highest: # 保存最优模型权重
                self.highest = top1_acc
                model.save_weights('sent_sim_amsoftmax.model')
            json.dump({'accs': self.accs, 'highest_top1': self.highest},
                      open('valid_amsoftmax.log', 'w'), indent=4)
            print ('top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc))

    evaluator = Evaluate()
    x_test = x_train[:1000]
    y_test = y_train[:1000]
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_data=(x_test, y_test),
                        #validation_split=0.05
                        )

    model.save_weights("gru_data/classfication_weight.h5")
    #model.predict(u"还有多少电")

if __name__=='__main__':
        # if os.path.exists('gru_data/data.config'):  # 如果有读取数据文件
        #     x_train,y_train,char2id,combined = pickle.load(open('gru_data/data.config', 'rb'))
        # else:
        x_train,y_train,char2id,combined = loadfile()
        # id与组别之间的映射
        train_cnn(x_train,y_train,char2id)



