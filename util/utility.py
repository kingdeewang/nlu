import math
import os
import regex as re
import warnings
import struct
from _functools import reduce
import requests
import json
from bs4 import BeautifulSoup
# pip3 install beautifulsoup4

import logging
import time
import random
from scipy.misc import logsumexp
from assertpy.assertpy import assert_that
import configparser
from DBUtils.PooledDB import PooledDB
from datetime import datetime
import traceback
import pickle
import sys

warnings.filterwarnings('ignore')

import numpy as np

workingDirectory = os.path.dirname(__file__) + '/../../'
modelsDirectory = workingDirectory + "models/"
corpusDirectory = workingDirectory + "corpus/"

lang = 'cn'


class Regex:

    def __init__(self, s, regex):
        self.iter = re.finditer(regex, s)

    def __iter__(self):  # Get iterator object on iter
        return self

    def __next__(self):
        return next(self.iter).groups()

    def groups(self):
        group = []
        for g in self:
            group.append(g)
        return group


class Text:

    def __init__(self, path, encoding="utf-8"):
        self.f = open(path, "r+" if os.path.isfile(path) else "w+", encoding=encoding)

    def __del__(self):
        self.f.close()

    def __iter__(self):  # Get iterator object on iter
        return self

    def __next__(self):
        while True:
            line = self.f.readline()
            if not line:
                self.home()
                raise StopIteration
            line = line.strip()

            if len(line) != 0:
                return line

    def prepend(self, s):
        '''
        prepend = append before the list;
        '''
        self.home()

        content = self.f.read()
        self.write(s)
#         self.f.seek(0, 2)
        self.f.write(content)
        self.f.flush()

    def append(self, s):
        self.end()

        if type(s) == str:
            self.f.write(s + '\n')
        else:
            for s in s:
                self.f.write(s + '\n')
        self.f.flush()

    def insert(self, index, s):
        if index < 0:
            self.end()
            index = -index - 1
            offset = self.f.tell() - 1
            while index > 0:
                index -= 1
                offset -= 1
                while True:
                    offset -= 1
                    _offset = self.f.seek(offset, os.SEEK_SET)
                    assert _offset == offset
                    char = self.f.read(1)
#                     print('char =', char)
                    if char[0] == '\n' or char[0] == '\r':
                        break
        else:
            self.home()
            offset = self.f.tell()
            while index > 0:
                index -= 1
                while True:
                    offset += 1
                    _offset = self.f.seek(offset, os.SEEK_SET)
                    assert _offset == offset
                    char = self.f.read(1)
#                     print('char =', char)
                    if char[0] == '\n' or char[0] == '\r':
                        break

        current_pos = self.f.tell()
        assert current_pos == offset + 1

        rest = self.f.read()
        self.f.seek(current_pos, os.SEEK_SET)

        if type(s) == str:
            self.f.write(s + '\n')
        else:
            for s in s:
                self.f.write(s + '\n')

        self.f.write(rest)

        self.f.flush()

    def __getitem__(self, index):
        if index < 0:
            self.end()
            index = -index
            offset = self.f.tell() - 1
            while index > 0:
                index -= 1
                offset -= 1
                while True:
                    offset -= 1
                    _offset = self.f.seek(offset, os.SEEK_SET)
                    assert _offset == offset
                    char = self.f.read(1)
    #                     print('char =', char)
                    if char == '\n' or char == '\r':
                        break
        else:
            self.home()
            offset = self.f.tell()
            while index > 0:
                index -= 1
                while True:
                    offset += 1
                    _offset = self.f.seek(offset, os.SEEK_SET)
                    assert _offset == offset
                    char = self.f.read(1)
#                     print('char =', char)
                    if char == '\n' or char == '\r':
                        break

        current_pos = self.f.tell()

        self.f.seek(current_pos, os.SEEK_SET)

        return self.f.readline().strip()

    def write(self, s):
        self.home()

        if type(s) == str:
            self.f.write(s + '\n')
        else:
            for s in s:
                self.f.write(str(s) + '\n')
        self.f.truncate()
        self.f.flush()

    def clear(self):
        self.home()
        self.f.write('')
        self.f.truncate()
        self.f.flush()

    def home(self):
        self.f.seek(0, os.SEEK_SET)

    def end(self):
        self.f.seek(0, os.SEEK_END)

    def collect(self):
        self.home()

        arr = []
        for line in self:
            if ord(line[0]) == 0xFEFF:
                line = line[1:]
                if not line:
                    continue
#             print(line)
            arr.append(line)
        return arr

    def remove(self, value):
        arr = self.collect()
        if value in arr:
            arr.remove(value)
            self.write(arr)

    def removeDuplicate(self):
        arr = []
        st = set()
        for s in self.collect():
            if s not in st:
                st.add(s)
                arr += [s]

        self.write(arr)


def discrepant_indices(x, y):
    index = []
    for i in range(len(y)):
        if x[i] != y[i]:
            index.append(i)
    return index


def errorMark(length, index):
    tag = ["  "] * length

    for i in index:
        tag[i] = "--"

    return tag


def convertWithAlignment(*arr):
    '''

    :param arr: an array of str array
    '''
    res = [""] * len(arr)

    size = len(arr[0])
    for j in range(size):
        l = [0] * len(arr)

        for i in range(len(arr)):
            res[i] += arr[i][j] + ' '
#             from util import utility
            l[i] = byte_length(arr[i][j])

        maxLength = max(l)
        for i in range(len(arr)):
            res[i] += ' ' * (maxLength - l[i])

    return res


def byte_length(value):
    length = 0
    for ch in value:
        if (ord(ch) & 0xff80) != 0:
            length += 2
        else:
            length += 1

    return length


def isConsecutive(seg):
    if len(seg) <= 1:
        return False

    for i in range(len(seg) - 1):
        if seg[i] != seg[i + 1]:
            return False
    return True


# convert fullwidth to Halfwidth
def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # space
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def containNaN(matrix):
    if type(matrix) not in (np.ndarray, list) :
        return matrix != matrix
    for x in matrix:
        if containNaN(x):
            return True
    return False


def validateNaN(matrix):
    if len(matrix.shape) == 1:
        for i in range(len(matrix)):
            if matrix[i] != matrix[i]:
                matrix[i] = 0;

    if type(matrix) != np.ndarray:
        return
    for x in matrix:
        validateNaN(x)


def string2id(char2id, s, maxlen=None):
    arr = [char2id.get(i, 1) for i in s]
    if maxlen:
        if len(arr) > maxlen:
            arr = arr[:maxlen]
        else :
            arr = arr + [0] * (maxlen - len(arr))
    return arr


def string2Index(char2id, s, maxlen=None):
    arr = [char2id.get(i, 0) for i in s]
    if maxlen:
        if len(arr) > maxlen:
            arr = arr[:maxlen]
        else :
            arr = arr + [-1] * (maxlen - len(arr))
    return arr


def readFolder(rootdir, sufix='.txt'):
    dic = {}

    for name in os.listdir(rootdir):
        path = os.path.join(rootdir, name)
        arr = []

        if path.endswith(sufix):
            arr = Text(path).collect()
            name = name[0:-len(sufix)]
        elif os.path.isdir(path):
            arr = readFolder_recursive(path, sufix)

        if not arr:
            continue

        if name not in dic:
            dic[name] = arr
        else:
            dic[name].extend(arr)

    return dic


def readFolder_recursive(rootdir, sufix='.txt'):
    dic = []

    for name in os.listdir(rootdir):
        path = os.path.join(rootdir, name)

        if path.endswith(sufix):
            dic.extend(Text(path).collect())
        elif os.path.isdir(path):
            dic.extend(readFolder_recursive(path, sufix))

    return dic


#      the output variable reaches its peak when the input variable becomes 1/2 ln3;
def sinusoidalHyperbolicTangentln3(x) :
    return math.sqrt(math.sin(math.tanh(x) * math.pi))


#      the output variable reaches its peak when the input variable becomes T;
def sinusoidalHyperbolicTangent(x, T):
    return sinusoidalHyperbolicTangentln3(0.5 * math.log(3) * x / T);

# coding=utf-8
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d


def conv2d_batch(input, filter, strides, padding=None):  # @ReservedAssignment @UnusedVariable
    ish = input.shape
    fsh = filter.shape
    output = np.zeros([ish[0], (ish[1] - fsh[0]) // strides[1] + 1, (ish[2] - fsh[1]) // strides[2] + 1, fsh[3]])
    osh = output.shape
    for p in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for di in range(fsh[0]):
                    for dj in range(fsh[1]):
                        t = np.dot(input[p, strides[1] * i + di, strides[2] * j + dj, :], filter[di, dj, :, :])
                        output[p, i, j] = np.sum([t, output[p, i, j]], axis=0)
    return output


def initial_offset(x, y, w, s, i):
    if y.shape[i] > 1:
        l = x.shape[i] + (w.shape[i] - s[i]) * (y.shape[i] - 1)
        if y.shape[i] * w.shape[i] < l:
            l = y.shape[i] * w.shape[i]
        return w.shape[i] - (2 * w.shape[i] + l - (l + w.shape[i] - 1) // w.shape[i] * w.shape[i] + 1) // 2
    else:
        return -((x.shape[i] - w.shape[i]) // 2)


# stride=(1, 1)
def conv2d_same(x, w, bias, s, activate):
#     assert x.dtype == w.dtype
    yshape0 = (x.shape[0] + s[0] - 1) // s[0]
    yshape1 = (x.shape[1] + s[1] - 1) // s[1]
    y = np.zeros([yshape0, yshape1, w.shape[3]], dtype=x.dtype)
    d0 = initial_offset(x, y, w, s, 0)
    d1 = initial_offset(x, y, w, s, 1)
    for i in range(yshape0):
        for j in range(yshape1):
            _i = s[0] * i - d0
            _j = s[1] * j - d1

            for di in range(max(0, -_i), min(w.shape[0], x.shape[0] - _i)):
                for dj in range(max(0, -_j), min(w.shape[1], x.shape[1] - _j)):
                    y[i, j] += np.dot(x[_i + di, _j + dj], w[di, dj])
            if bias is not None:
                y[i, j] += bias
            if activate is not None:
                y[i, j] = activate(y[i, j])
    return y


# stride=(1, 1)
def conv1d_same(x, w, bias, s, activate):
    yshape0 = (x.shape[0] + s - 1) // s
    y = np.zeros([yshape0, w.shape[2]], dtype=x.dtype)
    d0 = initial_offset(x, y, w, (s,), 0)
    for i in range(yshape0):
        _i = s * i - d0
        for di in range(max(0, -_i), min(w.shape[0], x.shape[0] - _i)):
            y[i] += np.dot(x[_i + di], w[di])
        if bias is not None:
            y[i] += bias
        if activate is not None:
            y[i] = activate(y[i])
    return y


# stride=(1, 1)
def conv2d_valid(x, w, bias, s, activate):
    yshape0 = (x.shape[0] + s[0] - w.shape[0]) // s[0]
    yshape1 = (x.shape[1] + s[1] - w.shape[1]) // s[1]

    y = np.zeros([yshape0, yshape1, w.shape[3]], dtype=x.dtype)
    for i in range(yshape0):
        for j in range(yshape1):
            for di in range(w.shape[0]):
                for dj in range(w.shape[1]):
                    y[i, j] += np.dot(x[s[0] * i + di, s[1] * j + dj], w[di, dj])
            if bias is not None:
                y[i, j] += bias
            if activate is not None:
                y[i, j] = activate(y[i, j])
    return y


def conv1d_valid(x, w, bias, s, activate):
    yshape0 = (x.shape[0] + s - w.shape[0]) // s

    y = np.zeros([yshape0, w.shape[2]], dtype=x.dtype)
    for i in range(yshape0):
        for di in range(w.shape[0]):
            y[i] += np.dot(x[s * i + di], w[di])
        if bias is not None:
            y[i] += bias
        if activate is not None:
            y[i] = activate(y[i])
    return y


# padding='VALID' or 'SAME'
def max_pooling(x, size=(2, 2), stride=(2, 2), padding='VALID'):
    # Preparing the output of the pooling operation.
    yshape0 = x.shape[0]
    yshape1 = x.shape[1]

    if padding == 'SAME':
        yshape0 += 1
        yshape1 += 1

    yshape0 //= stride[0]
    yshape1 //= stride[1]

    y = np.zeros((yshape0, yshape1, x.shape[2]), dtype=x.dtype)

    for i in range(yshape0):
        for j in range(yshape1):
            for k in range(y.shape[2]):
                _i = stride[0] * i
                _j = stride[1] * j
                y[i, j, k] = x[_i : _i + size[0], _j: _j + size[1], k].max()
    return y


def avg_pooling(x, size=(2, 2), stride=(2, 2), padding='VALID'):
    # Preparing the output of the pooling operation.
    yshape0 = x.shape[0]
    yshape1 = x.shape[1]

    if padding == 'SAME':
        yshape0 += 1
        yshape1 += 1

    yshape0 //= stride[0]
    yshape1 //= stride[1]

    y = np.zeros((yshape0, yshape1, x.shape[2]), dtype=x.dtype)

    for i in range(yshape0):
        for j in range(yshape1):
            for k in range(y.shape[2]):
                _i = stride[0] * i
                _j = stride[1] * j
                y[i, j, k] = x[_i : _i + size[0], _j: _j + size[1], k].mean()
    return y


def separable_conv2d(input, depth_filter, point_fileter, strides, rate=[1, 1], padding=None):  # @ReservedAssignment @UnusedVariable
    ish = input.shape
    fsh = depth_filter.shape
    psh = point_fileter.shape
    output = np.zeros([ish[0], (ish[1] - fsh[0]) // strides[1] + 1, (ish[2] - fsh[1]) // strides[2] + 1, psh[3]])
    osh = output.shape
    print(osh)
    for p in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for k in range(fsh[2]):
                    for di in range(fsh[0]):
                        for dj in range(fsh[1]):
                            t = np.dot(
                                    input[p, strides[1] * i + rate[0] * di, strides[2] * j + rate[1] * dj, k],
                                    depth_filter[di, dj, k, :]
                                )
                            t = np.dot(
                                    t,
                                    point_fileter[0, 0, k * fsh[3]:(k + 1) * fsh[3], :]
                                )
                            output[p, i, j, :] = np.sum(
                                    [
                                        t,
                                        output[p, i, j, :]
                                    ],
                                    axis=0
                                )
    return output


def depthwise_conv2d(input, filter, strides, rate=[1, 1], padding=None):  # @ReservedAssignment @UnusedVariable
    ish = input.shape
    fsh = filter.shape
    output = np.zeros([ish[0], (ish[1] - fsh[0]) // strides[1] + 1, (ish[2] - fsh[1]) // strides[2] + 1, ish[3] * fsh[3]])
    osh = output.shape
    print(osh)
    for p in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for k in range(fsh[2]):
                    for di in range(fsh[0]):
                        for dj in range(fsh[1]):
                            t = np.dot(
                                    filter[di, dj, k, :],
                                    input[p, strides[1] * i + rate[0] * di, strides[2] * j + rate[1] * dj, k],
                                )
                            output[p, i, j, k * fsh[3]:(k + 1) * fsh[3]] = np.sum(
                                    [
                                        t,
                                        output[p, i, j, k * fsh[3]:(k + 1) * fsh[3]]
                                    ],
                                    axis=0
                                )
    return output


def relu_inplace(feature_map):
    # Preparing the output of the ReLU activation function.
    for map_num in range(feature_map.shape[-1]):
        for r in range(feature_map.shape[0]):
            for c in range(feature_map.shape[1]):
                feature_map[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])


def relu(x):
    return x * (x > 0)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)


def linear(x):
    return x


def hard_sigmoid(x):
    x = 0.2 * x + 0.5
    x = x * (x >= 0.0) - 1.0
    return x * (x <= 0.0) + 1.0


def l2_normalize(x, axis=-1):
    l2 = np.linalg.norm(x, ord=2, axis=axis)
    return x / l2

# def l2_normalize(x):
#     y = x * x
#     y = y.sum(axix=-1)
#     z = y.sqrt()
#     return x / z


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sPunctuation = ",.:;!?()\\[\\]{}'\"=<>，。：；！？（）「」『』【】～‘’′”“《》、…．·"


def convertFromSegmentation(arr):
    s = ""
    for i in range(len(arr) - 1):
        s += arr[i]

        if arr[i][-1] in sPunctuation or arr[i + 1][0] in sPunctuation:
            continue
        s += " "

    s += arr[-1]
    return s


def convertToSegmentation(s):
    s = re.compile("([" + sPunctuation + "])").sub(' \\1 ', s)

    s = re.compile('(?<=[\\d])( +([\\.．：:]) +)(?=[\\d]+)').sub('\\2', s)

    while True:
        s, n = re.compile("([" + sPunctuation + "]) +\\1").subn('\\1\\1', s)
        if not n:
            break

    return s.strip().split()


def isEnglish(c):
    return re.compile('[a-zA-Zａ-ｚＡ-Ｚ\d]').match(c)


def convertToOriginal(arr):
    s = ""
    for i in range(len(arr) - 1):
        s += arr[i]

        if isEnglish(arr[i][-1]) and isEnglish(arr[i + 1][0]):
            s += " "

    s += arr[-1]
    return s


def equals(seg, segSub, I):
    i = I
    j = 0
    while i < len(seg) and j < len(segSub):
        if seg[i] != segSub[j] and not re.compile(segSub[j]).fullmatch(seg[i]):
            return False

        i += 1
        j += 1

    return True


def containsSubstr(seg, segSub, start=0):
    for i in range(start, len(seg) - len(segSub) + 1):
        if equals(seg, segSub, i):
            return i
    return -1;


class GRU:

    def __eq__(self, autre):
        return (self.Wxu == autre.Wxu).all() \
            and (self.Wxr == autre.Wxr).all() \
            and (self.Wxh == autre.Wxh).all()\
            and (self.Whu == autre.Whu).all()\
            and (self.Whr == autre.Whr).all()\
            and (self.Whh == autre.Whh).all()\
            and (self.bu == autre.bu).all()\
            and (self.br == autre.br).all()\
            and (self.bh == autre.bh).all()

    def __init__(self, *weight):

        def split(weight):
            shape = weight.shape
            embedSize = shape[-1] // 3

            return weight.reshape(shape[:-1] + (3, embedSize))

        if len(weight) == 3:
            self.Wxu, self.Wxr, self.Wxh = np.transpose(split(weight[0]), (1, 0, 2))

            self.Whu, self.Whr, self.Whh = np.transpose(split(weight[1]), (1, 0, 2))

            self.bu, self.br, self.bh = split(weight[2])
        else:
            self.Wxu, self.Wxr, self.Wxh, self.Whu, self.Whr, self.Whh, self.bu, self.br, self.bh = weight

        self.sigmoid = hard_sigmoid
        self.tanh = np.tanh

    def activate(self, x, h, mask=None, debug=None):
        r = self.sigmoid(x @ self.Wxr + h @ self.Whr + self.br)
        u = self.sigmoid(x @ self.Wxu + h @ self.Whu + self.bu)
        gh = self.tanh(x @ self.Wxh + (r * h) @ self.Whh + self.bh)

#         if debug is not None:
# debug.append(r.tolist())
#             debug.append(u.tolist())
#             debug.append(gh.tolist())

        _h = (1 - u) * gh + u * h
        if mask is None:
            return _h
        else:
            return np.where(np.expand_dims(mask, 1), _h, h)

    def call(self, x, return_sequences=False, mask=None, reverse=False, debug=None):
        if return_sequences:
            arr = np.zeros(x.shape, x.dtype)  # The desired data-type for the array, Default is numpy.float64.
        h = np.zeros(x.shape[0:-2] + x.shape[-1:], x.dtype)
        batch = len(x.shape) > 2

        if batch:
#         numpy.fliplr(mask) can also work here!
            for i in [*range(x.shape[1])][::-1] if reverse else range(x.shape[1]):
                h = self.activate(x[:, i], h, None if mask is None else mask[:, i])
                if return_sequences:
                    arr[:, i] = h
        else:
            for i in [*range(len(x))][::-1] if reverse else range(len(x)):
                if mask is not None and not mask[i]:
                    continue

                h = self.activate(x[i], h)
                if return_sequences:
                    arr[i] = h
            if debug is not None:
                debug.append(self.Wxu[0].tolist());
                debug.append(self.Wxr[0].tolist());
                debug.append(self.Wxh[0].tolist());
                debug.append(self.Whu[0].tolist());
                debug.append(self.Whr[0].tolist());
                debug.append(self.Whh[0].tolist());
                debug.append(self.bu.tolist())
                debug.append(self.br.tolist())
                debug.append(self.bh.tolist())

        return arr if return_sequences else h

    def write(self, file):
        writeArray(file, self.Wxu)
        writeArray(file, self.Wxr)
        writeArray(file, self.Wxh)

        writeArray(file, self.Whu)
        writeArray(file, self.Whr)
        writeArray(file, self.Whh)

        writeArray(file, self.bu)
        writeArray(file, self.br)
        writeArray(file, self.bh)

    @staticmethod
    def read(file):
        Wxu = readArray(file, 2)
        Wxr = readArray(file, 2)
        Wxh = readArray(file, 2)

        Whu = readArray(file, 2)
        Whr = readArray(file, 2)
        Whh = readArray(file, 2)

        bu = readArray(file, 1)
        br = readArray(file, 1)
        bh = readArray(file, 1)

        return GRU(Wxu, Wxr, Wxh, Whu, Whr, Whh, bu, br, bh)

    def weights(self):
        return np.concatenate([self.Wxu, self.Wxr, self.Wxh], axis=1), np.concatenate([self.Whu, self.Whr, self.Whh], axis=1), np.concatenate([self.bu, self.br, self.bh], axis=0)


class LSTM:

    def __eq__(self, autre):
        return (self.Wxi == autre.Wxi).all()\
            and (self.Wxf == autre.Wxf).all()\
            and (self.Wxc == autre.Wxc).all()\
            and (self.Wxo == autre.Wxo).all()\
            and (self.Whi == autre.Whi).all()\
            and (self.Whf == autre.Whf).all()\
            and (self.Whc == autre.Whc).all()\
            and (self.Who == autre.Who).all()\
            and (self.bi == autre.bi).all()\
            and (self.bf == autre.bf).all()\
            and (self.bc == autre.bc).all()\
            and (self.bo == autre.bo).all()

    def __init__(self, *weight):

        def split(weight):
            shape = weight.shape
            embedSize = shape[-1] // 4

            return weight.reshape(shape[:-1] + (4, embedSize))

        if len(weight) == 3:
            self.Wxi, self.Wxf, self.Wxc, self.Wxo = np.transpose(split(weight[0]), (1, 0, 2))

            self.Whi, self.Whf, self.Whc, self.Who = np.transpose(split(weight[1]), (1, 0, 2))

            self.bi, self.bf, self.bc, self.bo = split(weight[2])
        else:
            self.Wxi, self.Wxf, self.Wxc, self.Wxo, self.Whi, self.Whf, self.Whc, self.Who, self.bi, self.bf, self.bc, self.bo = weight

        self.sigmoid = hard_sigmoid
        self.tanh = np.tanh

    def activate(self, x, h, c, mask=None):
        i = self.sigmoid(x @ self.Wxi + h @ self.Whi + self.bi)
        f = self.sigmoid(x @ self.Wxf + h @ self.Whf + self.bf)
        _c = f * c + i * self.tanh(x @ self.Wxc + h @ self.Whc + self.bc)
        o = self.sigmoid(x @ self.Wxo + h @ self.Who + self.bo)

        _h = o * self.tanh(_c)
        if mask is None:
            return _h, _c
        else:
            return np.where(np.expand_dims(mask, 1), np.array([_h, _c]), np.array([h, c]))

#     // forward pass
    def call(self, x, return_sequences=False, mask=None, reverse=False, debug=None):
        if return_sequences:
            arr = np.zeros(x.shape, x.dtype)  # The desired data-type for the array, Default is numpy.float64.

        h = c = np.zeros(x.shape[0:-2] + x.shape[-1:], x.dtype)

        batch = len(x.shape) > 2

        if batch:
            for i in [*range(x.shape[1])][::-1] if reverse else range(x.shape[1]):
                h, c = self.activate(x[:, i], h, c, None if mask is None else mask[:, i])
                if return_sequences:
                    arr[:, i] = h
        else:
            for i in [*range(len(x))][::-1] if reverse else range(len(x)):
                if mask is not None and not mask[i]:
                    continue

                h, c = self.activate(x[i], h, c)
                if return_sequences:
                    arr[i] = h

        return arr if return_sequences else h

    def write(self, file):
        writeArray(file, self.Wxi)
        writeArray(file, self.Wxf)
        writeArray(file, self.Wxc)
        writeArray(file, self.Wxo)

        writeArray(file, self.Whi)
        writeArray(file, self.Whf)
        writeArray(file, self.Whc)
        writeArray(file, self.Who)

        writeArray(file, self.bi)
        writeArray(file, self.bf)
        writeArray(file, self.bc)
        writeArray(file, self.bo)

    @staticmethod
    def read(file):
        Wxi = readArray(file, 2)
        Wxf = readArray(file, 2)
        Wxc = readArray(file, 2)
        Wxo = readArray(file, 2)

        Whi = readArray(file, 2)
        Whf = readArray(file, 2)
        Whc = readArray(file, 2)
        Who = readArray(file, 2)

        bi = readArray(file, 1)
        bf = readArray(file, 1)
        bc = readArray(file, 1)
        bo = readArray(file, 1)

        return LSTM(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo)

    def weights(self):
        return np.concatenate([self.Wxi, self.Wxf, self.Wxc, self.Wxo], axis=1), np.concatenate([self.Whi, self.Whf, self.Whc, self.Who], axis=1), np.concatenate([self.bi, self.bf, self.bc, self.bo], axis=0)


class Bidirectional:

#     // forward pass
    def call(self, x, return_sequences=False, mask=None, debug=None):
        forward = self.forward.call(x, return_sequences, mask, debug=debug)
        backward = self.backward.call(x, return_sequences, mask, True, debug=debug)

        if debug is not None:
            debug.append([*forward])
            debug.append([*backward])

        if self.mode == 'sum':
            return forward + backward
        if self.mode == 'ave':
            return (forward + backward) / 2
        return np.concatenate([forward, backward], axis=-1)

    def write(self, file):
        self.forward.write(file)
        self.backward.write(file)

    def weights(self):
        return self.forward.weights() + self.backward.weights()


class BiLSTM(Bidirectional):

    def __eq__(self, autre):
        return self.forward == autre.forward and self.backward == self.backward and self.mode == autre.mode

    def __init__(self, weight, mode='concat'):
        length = len(weight) // 2

        self.forward = LSTM(*weight[:length])
        self.backward = LSTM(*weight[length:])
        self.mode = mode

    @staticmethod
    def read(file, mode='concat'):
        forward = LSTM.read(file)
        backward = LSTM.read(file)

        return BiLSTM(forward.weights() + backward.weights(), mode)


class BiGRU(Bidirectional):

    def __eq__(self, autre):
        return self.forward == autre.forward and self.backward == self.backward and self.mode == autre.mode

    def __init__(self, weight, mode='concat'):
        length = len(weight) // 2
        self.forward = GRU(*weight[:length])
        self.backward = GRU(*weight[length:])
        self.mode = mode

    @staticmethod
    def read(file, mode='concat'):
        forward = GRU.read(file)
        backward = GRU.read(file)
        return BiGRU(forward.weights() + backward.weights(), mode)


class CRF:

    def __eq__(self, autre):
        return (self.bias == autre.bias).all()\
            and (self.G == autre.G).all()\
            and (self.kernel == autre.kernel).all()\
            and (self.left_boundary == autre.left_boundary).all()\
            and (self.right_boundary == autre.right_boundary).all()

# G has been transposed!
    def __init__(self, kernel, G, bias, left_boundary, right_boundary):
        self.kernel = kernel
        self.G = G
        self.bias = bias
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

        self.activation = linear

    def viterbi_one_hot(self, X, mask=None):
        return np.eye(len(self.bias), dtype=int)[self.call(X, mask=mask)]

    def shifting_matrix(self, mask, left=True):
# transform leading padding into tailing padding to facilitate forward calculation
# suppose we want to transform the following leading-padded matrix into tailing-padded matrix
# 0 0 0 0 1 2 3 4 5 6 7 8 9
#                ||
#                \/
# 1 2 3 4 5 6 7 8 9 0 0 0 0
# for index = 4
#     [0 0 0 0 1 0 0 0 0 0 0 0 0]   [0]
#     [0 0 0 0 0 1 0 0 0 0 0 0 0]   [0]
#     [0 0 0 0 0 0 1 0 0 0 0 0 0]   [0]
#     [0 0 0 0 0 0 0 1 0 0 0 0 0]   [0]
#     [0 0 0 0 0 0 0 0 1 0 0 0 0]   [1]
#     [0 0 0 0 0 0 0 0 0 1 0 0 0]   [2]
#     [0 0 0 0 0 0 0 0 0 0 1 0 0]   [3]
#     [0 0 0 0 0 0 0 0 0 0 0 1 0] * [4] = [1 2 3 4 5 6 7 8 9 0 0 0 0]
#     [0 0 0 0 0 0 0 0 0 0 0 0 1]   [5]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [6]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [7]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [8]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [9]

# transform tailing padding into leading padding to facilitate backward calculation
# suppose we want to transform the following tailing-padded matrix into leading-padded matrix
# 1 2 3 4 5 6 7 8 9 0 0 0 0
#                ||
#                \/
# 0 0 0 0 1 2 3 4 5 6 7 8 9
# for index = 4
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [1]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [2]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [3]
#     [0 0 0 0 0 0 0 0 0 0 0 0 0]   [4]
#     [1 0 0 0 0 0 0 0 0 0 0 0 0]   [5]
#     [0 1 0 0 0 0 0 0 0 0 0 0 0]   [6]
#     [0 0 1 0 0 0 0 0 0 0 0 0 0]   [7]
#     [0 0 0 1 0 0 0 0 0 0 0 0 0] * [8] = [0 0 0 0 1 2 3 4 5 6 7 8 9]
#     [0 0 0 0 1 0 0 0 0 0 0 0 0]   [9]
#     [0 0 0 0 0 1 0 0 0 0 0 0 0]   [0]
#     [0 0 0 0 0 0 1 0 0 0 0 0 0]   [0]
#     [0 0 0 0 0 0 0 1 0 0 0 0 0]   [0]
#     [0 0 0 0 0 0 0 0 1 0 0 0 0]   [0]

        length = mask.shape[-1]

        arange = np.arange(0, length)

        index = np.sum(1 - mask.astype(np.int32), -1)

        if left:
            matrix = np.expand_dims(arange - np.expand_dims(index, -1), -2) - np.expand_dims(arange, -1) == 0
            mask = arange < np.expand_dims(length - index, -1)
        else:
            matrix = np.expand_dims(arange - np.expand_dims(index, -1), -1) - np.expand_dims(arange, 0) == 0
            mask = arange >= np.expand_dims(index, -1)
        return mask, matrix

    def format_input(self, mask, *arr):
        batch = len(mask.shape) > 1

        if batch and not mask[:, 0].all() or not batch and not mask[0]:
            mask, shifting_matrix = self.shifting_matrix(mask)
            arr = [np.matmul(shifting_matrix.astype(x.dtype), x) for x in arr]

        return (mask, *arr)

    def add_left_boundary(self, x):
        batch = len(x.shape) > 2
        if batch:
            x[:, 0] += self.left_boundary
        else:
            x[0] += self.left_boundary
        return x

    def add_right_boundary(self, x, mask):
        batch = len(x.shape) > 2
        if batch:
            end = np.expand_dims(np.expand_dims(self.right_boundary, 0), 0)

            if mask is None:
                x[:, -1] += self.right_boundary
            else:
                mask = np.expand_dims(mask.astype(np.float32), -1)
                zeros = np.zeros((np.shape(mask)[0], 1, 1), np.float32)

    #         shift left by offset = 1:
    #         0 0 1 1 1 1 1 1 1 1 1 0 0 0 0------mask
    #         0 1 1 1 1 1 1 1 1 1 0 0 0 0 0------mask_
    #                             ^
                end_mask = (mask > np.concatenate([mask[:, 1:], zeros], axis=1)).astype(np.float32)
                x += end_mask * end
        else:
            if mask is None:
                x[-1] += self.right_boundary
            else:
                end = np.expand_dims(self.right_boundary, 0)

                zeros = np.zeros((1,), x.dtype)
    #         shift left by offset = 1:
    #         0 0 1 1 1 1 1 1 1 1 1 0 0 0 0
    #         0 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    #                             ^
                end_mask = mask > np.concatenate([mask[1:], zeros], axis=0)
                x += np.expand_dims(end_mask, -1) * end

        return x

    def call(self, x, mask=None, mask_pos=None):
        batch = len(x.shape) > 2
        x = x @ self.kernel + self.bias

        mask, x = self.format_input(mask, x)
        x = self.add_left_boundary(x)
        x = self.add_right_boundary(x, mask)

        if batch:
            argmin_tables = np.zeros(x.shape, np.int32)
            min_energy = x[:, 0]
            length = x.shape[1]
            G = np.expand_dims(self.G, 0)
            if mask_pos is not None:
                mask_pos = np.expand_dims(mask_pos, 2)
            out_of_bound = -np.ones(min_energy.shape, np.int)
            if mask is not None:
                mask = np.expand_dims(mask, -1)
            for i in range(length - 1):
                energy = G + np.expand_dims(min_energy, 1)
# performance tip: numpy.argmin(-1) executes much faster than numpy.argmin(axis) when axis is other than -1!
                argmin = (energy if mask_pos is None else energy + (1 - mask_pos[:, i]) * 1e10).argmin(-1)

                if mask is None:
                    argmin_tables[:, i] = argmin
                    min_energy = x[:, i + 1] + np.min(energy, -1)
                else:
                    argmin_tables[:, i] = np.where(mask[:, i], argmin, out_of_bound)
                    min_energy = np.where(mask[:, i + 1], x[:, i + 1] + np.min(energy, -1), min_energy)

            argmin = np.expand_dims(np.argmin(min_energy, -1), -1)

            if mask is not None:
                mask, shifting_matrix = self.shifting_matrix(np.squeeze(mask, -1), False)
                argmin_tables = np.matmul(shifting_matrix.astype(np.int32), argmin_tables + 1) - 1
    #             assumming that mask[:, -1] is all true
                assert mask[:, -1].all()
                mask = np.expand_dims(mask, -1)

            best_paths = np.zeros(x.shape[:-1], np.int32)
            out_of_bound = out_of_bound[:, :1]

            best_paths[:, -1] = argmin.squeeze()

            for i in range(-2, -length - 1, -1):
#                 _argmin may contain negatives, but argmin is perfectly non-negative!
                _argmin = np.take_along_axis(argmin_tables[:, i], argmin, axis=1)
                if mask is None:
                    argmin = _argmin
                    best_paths[:, i] = argmin.squeeze()
                else:
                    mask_i = mask[:, i]
                    argmin = np.where(mask_i, _argmin, argmin)
                    best_paths[:, i] = np.where(mask_i, _argmin, out_of_bound).squeeze()

        else:
            argmin_tables = np.zeros(x.shape, np.int32)
            min_energy = x[0]
            length = len(x)
            for i in range(length - 1):
                energy = self.G + min_energy

                argmin = (energy if mask_pos is None else energy + (1 - mask_pos[i]) * 1e10).argmin(-1)

                if mask is None:
                    argmin_tables[i] = argmin
                    min_energy = x[i + 1] + np.min(energy, -1)
                else:
                    argmin_tables[i] = argmin if mask[i] else -1

                    if mask[i + 1]:
                        min_energy = x[i + 1] + np.min(energy, -1)

            if mask_pos is not None:
                min_energy += (1 - mask_pos[-1]) * 1e10
            argmin = np.argmin(min_energy, -1)

            if mask is not None:
                mask, shifting_matrix = self.shifting_matrix(mask, False)
                argmin_tables = np.matmul(shifting_matrix.astype(np.int32), argmin_tables + 1) - 1
                assert mask[-1]

            best_paths = -np.ones(length, np.int32)

            best_paths[-1] = argmin

            for i in range(-2, -length - 1, -1):
                if mask is None or mask[i]:
                    argmin = argmin_tables[i][argmin]
                    best_paths[i] = argmin
        return best_paths

    def loss(self, x, y_true, mask):
        x = self.activation(np.dot(x, self.kernel) + self.bias)
        mask, x, y_true = self.format_input(mask, x, np.expand_dims(y_true, -1))
        y_true = np.squeeze(y_true, -1)

        x = self.add_left_boundary(x)
        x = self.add_right_boundary(x, mask)

        input_score = np.squeeze(np.take_along_axis(x, np.expand_dims(y_true, -1), -1), -1)
        G = self.G

        batch = len(x.shape) > 2
        if batch:
            chain_energy = np.dot(np.eye(*G.shape)[y_true][:, 1:], G)

            _chain_energy = np.squeeze(np.take_along_axis(chain_energy, np.expand_dims(y_true[:, :-1], -1), -1), -1)
            assert_that(all([chain_energy[i, j, y_true[i, :-1][j]] == _chain_energy [i][j] for i in range(len(x)) for j in range(chain_energy.shape[1])])).is_true()
            chain_energy = _chain_energy
        else:
            chain_energy = np.dot(np.eye(*G.shape)[y_true][1:], G)

            _chain_energy = np.squeeze(np.take_along_axis(chain_energy, np.expand_dims(y_true[:-1], -1), -1), -1)
            assert_that(all([chain_energy[j, y_true[:-1][j]] == _chain_energy[j] for j in range(chain_energy.shape[0])])).is_true()
            chain_energy = _chain_energy

        if mask is not None:
            mask_f = mask.astype(np.float32)
            input_score *= mask_f

            if batch:
                chain_energy *= mask_f[:, 1:] * mask_f[:, :-1]
            else:
                chain_energy *= mask_f[1:] * mask_f[:-1]

            mask = np.expand_dims(mask, -1)
        energy = np.sum(input_score, -1) + np.sum(chain_energy, -1)

#         return energy
        length = np.shape(x)[-2]
        if batch:
            G = np.expand_dims(G, 0)
            minus_x = -x[:, 0]
        else:
            minus_x = -x[0]

        for i in range(1, length):
            if batch:
                _minus_x = logsumexp(np.expand_dims(minus_x, -2) - G, -1) - x[:, i]
                if mask is None:
                    minus_x = _minus_x
                else:
                    minus_x = np.where(mask[:, i], _minus_x, minus_x)
            else:
                _minus_x = logsumexp(np.expand_dims(minus_x, -2) - G, -1) - x[i]
                if mask is None:
                    minus_x = _minus_x
                else:
                    minus_x = np.where(mask[i], _minus_x, minus_x)

        mask = np.squeeze(mask)
        nloglik = logsumexp(minus_x, -1) + energy
        nloglik /= length if mask is None else np.sum(mask, -1)
        return nloglik

    def write(self, file):
        writeArray(file, self.kernel)
        writeArray(file, self.G)
        writeArray(file, self.bias)
        writeArray(file, self.left_boundary)
        writeArray(file, self.right_boundary)

    @staticmethod
    def read(file):
        kernel = readArray(file, 2)
        G = readArray(file, 2)
        bias = readArray(file, 1)
        left_boundary = readArray(file, 1)
        right_boundary = readArray(file, 1)

        return CRF(kernel, G, bias, left_boundary, right_boundary)

    def weights(self):
        return self.kernel, self.G, self.bias, self.left_boundary, self.right_boundary


class Embedding:

    def __eq__(self, autre):
        return self.char2id == autre.char2id and (self.wEmbedding == autre.wEmbedding).all()

    def __init__(self, char2id, wEmbedding):
        self.char2id = char2id
        self.wEmbedding = wEmbedding

    def call(self, sent, mask_zero=True, max_length=None):
        if max_length:
            sent = sent[:max_length]
        if self.char2id is None:
            assert isinstance(sent[0], int)
            index = np.array(sent)
        elif isinstance(sent, (list, tuple)):
            if isinstance(sent[0], (list, tuple)):
                index = [[[self.char2id.get(c, 1) for c in w] for w in s] for s in sent]  # @UndefinedVariable
                index = format_data(index)
            else:
                index = [[self.char2id.get(c, 1) for c in w] for w in sent]
                index = format_data([index])[0]
        else :
            index = np.array([self.char2id.get(c, 1) for c in sent])

        if mask_zero:
            return self.wEmbedding[index], index != 0
        else:
            return self.wEmbedding[index]

    def write(self, file):
        if self.char2id:
            writeCharDict(file, self.char2id)
        writeArray(file, self.wEmbedding)

    @staticmethod
    def read(file, dic=True):
        if dic:
            char2id = readCharDict(file)
        else:
            char2id = None

        wEmbedding = readArray(file, 2)

        return Embedding(char2id, wEmbedding)


class Dense:

    def __eq__(self, autre):
        return (self.wDense == autre.wDense).all() and (self.bDense == autre.bDense).all()

    def __init__(self, wDense, bDense=None):
        self.wDense = wDense
        self.bDense = bDense

    def call(self, x):
        x = np.matmul(x, self.wDense)
        if self.bDense is not None:
            x += self.bDense
        return x

    def write(self, file):
        writeArray(file, self.wDense)
        if self.bDense is not None:
            writeArray(file, self.bDense)

    @staticmethod
    def read(file, bias=True):
        wDense = readArray(file, 2)
        if bias:
            bDense = readArray(file, 1)
        else:
            bDense = None

        return Dense(wDense, bDense)


class Conv2D:

    def __eq__(self, autre):
        return (self.wCNN == autre.wCNN).all() and (self.bCNN == autre.bCNN).all()

    def __init__(self, wCNN, bCNN):
        self.wCNN = wCNN
        self.bCNN = bCNN
        self.paddingSame = False

    def call(self, x):
        return conv2d_same(x, self.wCNN, self.bCNN, (1, 1), relu) if self.paddingSame else conv2d_valid(x, self.wCNN, self.bCNN, (1, 1), relu)

    def write(self, file):
        writeArray(file, self.wCNN)
        writeArray(file, self.bCNN)

    @staticmethod
    def read(file):
        wDense = readArray(file, 3)
        bDense = readArray(file, 1)

        return Conv2D(wDense, bDense)


class Conv1D:

    def __eq__(self, autre):
        return (self.wCNN == autre.wCNN).all() and (self.bCNN == autre.bCNN).all()

    def __init__(self, wCNN, bCNN):
        self.wCNN = wCNN
        self.bCNN = bCNN
        self.paddingSame = True

    def call(self, x, mask=None):
        x *= np.expand_dims(mask, -1)
        return conv1d_same(x, self.wCNN, self.bCNN, 1, relu) if self.paddingSame else conv1d_valid(x, self.wCNN, self.bCNN, 1, relu)

    def write(self, file):
        writeArray(file, self.wCNN)
        writeArray(file, self.bCNN)

    @staticmethod
    def read(file):
        wDense = readArray(file, 3)
        bDense = readArray(file, 1)

        return Conv1D(wDense, bDense)


def writeArray(file, array):
    if len(array.shape) == 1:
        file.write(struct.pack('>i', *array.shape))
        file.write(struct.pack('>' + str(array.size) + 'd', *array))
    else:
        file.write(struct.pack('>' + str(len(array.shape)) + 'i', *array.shape))
        file.write(struct.pack('>' + str(array.size) + 'd', *array.reshape(-1)))


def readArray(file, dimension):
    if dimension > 1:
        fmt = '>' + str(dimension) + 'i'

        dimension = struct.unpack(fmt, file.read(struct.calcsize(fmt)))

        fmt = '>' + str(reduce(lambda x, y: x * y, dimension)) + 'd'
        arr = struct.unpack(fmt, file.read(struct.calcsize(fmt)))
        return np.array(arr).reshape(dimension)
    else:
        fmt = '>i'

        dimension = struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]

        fmt = '>' + str(dimension) + 'd'
        arr = struct.unpack(fmt, file.read(struct.calcsize(fmt)))
        return np.array(arr)


def writeCharDict(file, char2id):
    file.write(struct.pack('>i', len(char2id)))
    for key, value in char2id.items():
        file.write(struct.pack('>H', ord(key)))
        file.write(struct.pack('>i', value))


def readCharDict(file):
    char2id = {}
    length = struct.unpack('>i', file.read(struct.calcsize('i')))[0]

    for _ in range(length):
        fmt = '>H'

        c = struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]

        value = struct.unpack('>i', file.read(struct.calcsize('i')))[0]
        char2id[chr(c)] = value
    return char2id


def writeWordDict(file, word2id):
    file.write(struct.pack('>i', len(word2id)))
    for key, value in word2id.items():
        file.write(struct.pack('>i', len(key)))
        file.write(struct.pack('>' + str(len(key)) + 'H', *[ord(x) for x in key]))
        file.write(struct.pack('>i', value))


def readWordDict(file):
    word2id = {}
    length = struct.unpack('>i', file.read(struct.calcsize('i')))[0]

    for _ in range(length):
        l = struct.unpack('>i', file.read(struct.calcsize('i')))[0]
        fmt = '>' + str(l) + 'H'

        s = ''
        for c in struct.unpack(fmt, file.read(struct.calcsize(fmt))):
            s += chr(c)

        value = struct.unpack('>i', file.read(struct.calcsize('i')))[0]
        word2id[s] = value
    return word2id


def reshape(_matrix, matrix):
    i = 0
    _matrix = list(_matrix)
    newMatrix = []
    for v in matrix:

        _v = np.array(_matrix[i:i + v.size]).reshape(v.shape)
        newMatrix.append(_v)

        i += v.size
    assert i == len(_matrix)
    return newMatrix


def getRecords(url):

    wb_data_sub = requests.get(url)

    soup = BeautifulSoup(wb_data_sub.text, 'lxml')

    data = soup.p.string

    text = json.loads(data)

    records = text['response']['docs']

    return records


class Comment:

    def __init__(self, s, parent=None):
        self.kinder = [s]
        self.parent = parent
        if parent:
            parent.kinder.append(self)


def iscomment(s):
    s = s.strip()
    if not s:
        return False

    if s[0] == '#':
        return True

    return False


def isdocstring(s):
    s = s.strip()
    if not s:
        return False

    if s.startswith('"""'):
        if s.endswith('"""'):
            return True
    return False


class Struct:

    def __init__(self, s, indent=0, parent=None):
        self.kinder = [s]
        self.indent = indent
        self.parent = parent
        if parent:
            parent.kinder.append(self)

    def shift(self, offset):
        self.indent += offset
        for i in range(len(self.kinder)):
            if isinstance(self.kinder[i], str):
                if iscomment(self.kinder[i]):
                    s = self.kinder[i].strip()
                    self.kinder[i] = s
                else:
                    self.kinder[i] = ' ' * offset + self.kinder[i]
            else :
                self.kinder[i].shift(offset)

    def encounter(self, s):
        m = re.compile('\S').search(s)
        if not m:
            self.kinder.append(s)
            print('empty line detected!')
            return self
        indent = m.start()

        if iscomment(s):
            print('comment line detected!')
            self.kinder.append(s)
#         elif len(self.kinder[-1]) > 0 and (self.kinder[-1][-1] in ('{', '(', ',') or self.kinder[-1][-1] == s[indent] and s[indent] in ('"', "'")):
#             print('line continuation detected!')
#             self.kinder.append(s)
        elif indent == self.indent:
            print('code line of the same indent detected!')
            self.kinder.append(s)
        elif self.indent < indent :
            print('indentation detected!')
            self = Struct(s, indent, self)
        else:
            while True:
                self = self.parent
                if indent == self.indent:
                    self.kinder.append(s)
                    break
                if indent > self.indent:
                    self = Struct(s, indent, self)
                    break

        return self

    def string_array(self):
        arr = []
        for child in self.kinder:
            if isinstance(child, str):
                arr.append(child)
            else:
                arr += child.string_array()
        return arr

    def __str__(self):
        return '\n'.join(self.string_array())

    def __repr__(self):
        return '\n'.join(self.string_array())

    def validate(self, indent=0):
        self.indent = indent
        for i in range(len(self.kinder)):
            struct = self.kinder[i]
            if isinstance(struct, str):
                if iscomment(struct):
                    s = struct.strip()
                    self.kinder[i] = s
            else:
                print(struct)
                print('indent = ', struct.indent)
                if struct.indent < indent + 4:
                    struct.shift(indent + 4 - struct.indent)
                    print('after shifting')
                    print(struct)
                struct.validate(indent + 4)


def substitute(regex, substitution, s):
    if re.compile(regex).search(s):
        return re.compile(regex).sub(substitution, s)
    return s


def translateJava(file, python_file=None):
    if python_file is None:
        python_file = file

    arr = []
    with open(file, 'r', encoding='utf8') as file:
        for s in file.readlines():
            s = s.rstrip()
            print(s)

            string = []
            buffer = []
            start = 0
            for m in re.compile('(?<!\\\)"[^"]*(?<!\\\)"').finditer(s):
                buffer.append(s[start: m.start()])
                buffer.append('java_string')
                string.append(m.group(0))
                start = m.end()
            if 0 < start < len(s):
                buffer.append(s[start:])
            if buffer:
                s = ''.join(buffer)

            if re.compile('\s*#').match(s) or re.compile('package\s+[\w+\.]+;').fullmatch(s) or re.compile('import\s+[\w+\.]+;').fullmatch(s):
                continue

            if s.strip() == '@Override':
                continue
# set definition in python
#             s = substitute('\{(\s*[^{,]+(?:,\s*[^,}]+)*)\}', '[\\1]', s)

            if s.endswith(';'):
                s = s[:-1]

            s = substitute('(^[^{]*)\}$', '\\1', s)

            s = substitute('(.+)\{$', '\\1:', s)
            s = substitute('(.+)\{#', '\\1: #', s)

            if s.endswith('else'):
                s += ':'

            s = substitute('if\s*\(([\s\S]+?)\)$', 'if \\1:', s)

            m = re.compile('(\s+if)\s*\(([\s\S]+?)\)\s*:').fullmatch(s)
            if m:
                s = m.group(1) + ' ' + m.group(2) + ':'

            s = substitute('(if|while)\s*\((.+)\):$', '\\1 \\2:', s)
            s = substitute('[\w\.]+(?<!def|not)\s+(\w+)\(\):', 'def \\1(self):', s)

            s = substitute('[\w\.]+(?<!(?:^|\W)(?:def|class|if|in))\s+(\w+)\s*\((?:\w+\s+)?(\w+)\):', 'def \\1(self, \\2):', s)
            s = substitute('[\w\.]+(?<!(?:^|\W)(?:def|class|if|in))\s+(\w+)\s*\((?:\w+\s+)?(\w+),\s*(?:\w+\s+)?(\w+)\):', 'def \\1(self, \\2, \\3):', s)
            s = substitute('[\w\.]+(?<!(?:^|\W)(?:def|class|if|in))\s+(\w+)\s*\((\w+(?:,\s*(\w+))+)\):', 'def \\1(self, \\2):', s)

            s = substitute('[\w\.]+\s+(\w+)\(\w+\s+(\w+),\s*\w+\s+(\w+),\s*\w+\s+(\w+)\)\s*:', 'def \\1(self, \\2, \\3, \\4):', s)

            s = substitute('(.)\s+:', '\\1:', s)
            s = substitute('log\.info\(', 'print(', s)

            s = substitute('(ArrayList|Vector)\(\)', '[]', s)
            s = substitute('(?<!\w)(Hash|Tree)Set(?!\w)', 'set', s)
            s = substitute('(?<!\w)(Hash|Tree)Map\(\)', '{}', s)

            s = substitute('(?<!\w)false(?!\w)', 'False', s)
            s = substitute('(?<!\w)true(?!\w)', 'True', s)

            s = substitute('([\w\.]+)\[\]', '\\1', s)

            s = substitute('(def\s+(\w+)\(\))$', '\\1:', s)

            s = substitute('\w+\s+(\w+)\s*=(?!=)([\s\S]+)', '\\1 =\\2', s)

            s = substitute('new\s+[\w\.]+\[([\w\. \+\-]+)\](?:\[\])*', '[None] * (\\1)', s)
            s = substitute('new\s+(\w+)', '\\1', s)
            s = substitute('\(\s*\w+(?<!not|or|in)\s+(?!not|or|in)(\w+)', '(\\1', s)
            s = substitute(',\s*\w+\s+(?!in\W)(\w+)', ', \\1', s)

            s = substitute('}\s*el', 'el', s)

            s = substitute('System\.out\.println', 'print', s)
            s = substitute('(?<!\w)throw(?!\w)', 'raise', s)
            s = substitute('([\w\.]+)\.length\(\)', 'len(\\1)', s)
            s = substitute('([\w\.]+)\.length(?=[^\w\(])', 'len(\\1)', s)
            s = substitute('([\w\.\]\[]+)\.next\(\)', 'next(\\1)', s)

            s = substitute('((?!\.)[\w\.]+)\.size\(\)', 'len(\\1)', s)

            s = substitute('([\w\.]+)\.charAt\(([\w\.]+)\)', '\\1[\\2]', s)

            s = substitute('(?<!\w)null(?!\w)', 'None', s)

            s = substitute('\s*extends\s+([\w\.]+):', '(\\1):', s)
            s = substitute('}\s*finally:', 'finally:', s)

            s = substitute('\s*implements\s+([\w\.]+(?:,\s*[\w\.]+)?):', '(\\1):', s)

            s = substitute('for\s*\(\s*;\s*;\s*\)', 'while True', s)
            s = substitute('for\s*\(\s*;\s*([^;]*?)\s*;\s*\)', 'while \\1', s)

            s = substitute('switch\s*\(([\w\(\)\[\]\.]+)\)\s*:', 'switch_x = \\1', s)

            s = substitute('(?<!\w)default:', 'else:', s)
            s = substitute('(?<!\w)(public|static|final|transient|private|protected|abstract)\s+', '', s)
            s = substitute('(?<!\w)this(?!\w)', 'self', s)
            s = substitute('(?<!\w)toString(?!\w)', '__str__', s)
            s = substitute('(?<!\w)hashCode(?!\w)', '__hash__', s)

            s = substitute('(?<!\w)indexOf(?!\w)', 'index', s)
            s = substitute('\.trim\(\)', '.strip()', s)
            s = substitute('\s*&&\s*', ' and ', s)
            s = substitute('!\s*(?!=)', 'not ', s)
            s = substitute('\s*\|\|\s*', ' or ', s)
            s = substitute('else if', 'elif', s)
            s = substitute('/\*', '"""', s)
            s = substitute(' \*/', '"""', s)
            s = substitute('\*/', '"""', s)
            s = substitute('//', '#', s)

            s = substitute('(?<!\w)[\w\.]+\.\.\.\s*', '', s)

            s = substitute('(?<!\w)startsWith(?!\w)', 'startswith', s)
            s = substitute('(?<!\w)toUpperCase(?!\w)', 'upper', s)
            s = substitute('(?<!\w)toLowerCase(?!\w)', 'lower', s)

            s = substitute('(?<!\w)RuntimeException(?!\w)', 'Exception', s)

            s = substitute('(?<!\w)endsWith(?!\w)', 'endswith', s)
            s = substitute('(?<!\w)super\.', 'super(type(self), self).', s)
            s = substitute('(?<!\w)super\((\w+(\s*,\s*\w+)*)\)$', 'super(type(self), self).__init__(\\1)', s)

            s = substitute("case\s+(\S+):", 'elif switch_x == \\1:', s)

            s = substitute("([\w\.]+)\s+instanceof\s+([\w\.]+)", 'isinstance(\\1, \\2)', s)

            s = substitute("([\w\.]+)\.substring\s*\(([\w\.]+),\s*([\w\.]+)\)", '\\1[\\2:\\3]', s)

            s = substitute('([\w\.]+)\.substring\s*\(([\w\.]+)\)', '\\1[\\2:]', s)
            s = substitute('\.set\((\w+),\s*(\w+)\)', '[\\1] = \\2', s)

            s = substitute('<[\w\.]*>', '', s)
            s = substitute('<[\w\.]+(,\s*[\w\.]+)?>', '', s)
            s = substitute('^\s*(?!(return|raise|import|yield|assert))\w+\s+\w+(,\s*\w+)*$', '', s)

            s = substitute('\.get\(([\w\-\+ ]+)\)', '[\\1]', s)
            s = substitute('\.put\(([^,]+), *(.+)\)$', '[\\1] = \\2', s)
            s = substitute('\.substring\(([^,]+)\)', '[\\1:]', s)
            s = substitute('\.substring\(([^,]+),\s*([^()]+)\)', '[\\1:\\2]', s)

            s = substitute('^(\s+)\w+\((\w+(?:,\s*\w+)*)\):', '\\1def __init__(self, \\2):', s)
            s = substitute('^(\s+)\w+\(\):', '\\1def __init__(self):', s)

            s = substitute('for\s*\((\w+)\s*=\s*([^;]+)\s*;\s*\\1\s*<\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:\+\+\\1|\\1\+\+)\):', 'for \\1 in range(\\2, \\3):', s)
            s = substitute('for\s*\((\w+)\s*=\s*([^;]+)\s*;\s*\\1\s*<=\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:\+\+\\1|\\1\+\+)\):', 'for \\1 in range(\\2, \\3 + 1):', s)

            s = substitute('for\s*\(\s*;\s*(\w+)\s*<\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:\+\+\\1|\\1\+\+)\):', 'for \\1 in range(\\1, \\2):', s)
            s = substitute('for\s*\(\s*;\s*(\w+)\s*<=\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:\+\+\\1|\\1\+\+)\):', 'for \\1 in range(\\1, \\2 + 1):', s)

            s = substitute('for\s*\((\w+)\s*=\s*([^;]+)\s*;\s*\\1\s*>\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:--\\1|\\1--)\):', 'for \\1 in range(\\2, \\3, -1):', s)
            s = substitute('for\s*\((\w+)\s*=\s*([^;]+)\s*;\s*\\1\s*>=\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:--\\1|\\1--)\):', 'for \\1 in range(\\2, \\3 - 1, -1):', s)

            s = substitute('for\s*\(\s*;\s*(\w+)\s*>\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:--\\1|\\1--)\):', 'for \\1 in range(\\1, \\2, -1):', s)
            s = substitute('for\s*\(\s*;\s*(\w+)\s*>=\s*([\w\(\)\[\]\.\-\+ ]+)\s*;\s*(?:--\\1|\\1--)\):', 'for \\1 in range(\\1, \\2 - 1, -1):', s)

            s = substitute('for\s*\((?:[\w\.]+\s+)?(\w+)\s*:\s*([\s\S]+?)\)(?::)?$', 'for \\1 in \\2:', s)

            s = substitute('([^=]+)=([^\?]+)\?([^:]+):(.+)', '\\1=\\3 if \\2 else \\4', s)
            s = substitute('(?<=return *)([^\?]+)\?([^:]+):(.+)', '\\2 if \\1 else \\3', s)

            s = substitute('(?<!\w)throws\s+.*:$*', ':', s)
            s = substitute('(\w+)\s{2,}(\w+)', '\\1 \\2', s)

            s = substitute('^(\s*)\+\+(.+)', '\\1\\2 += 1', s)

            s = substitute('}\s*catch\s*\((\w+)\):', 'except Exception as \\1:', s)

            s = substitute('\s*\.equals\(([^),]+)\)', ' == \\1', s)

            s = substitute('([\w\.]+)\.contains(?:Key)?\(([\w\.]+)\)', '\\2 in \\1', s)

            s = substitute('([\w]+)\.printStackTrace\(\)', 'print(\\1)', s)
            s = substitute('\.addAll\(', '.extend(', s)

            s = substitute('([\w\.]+)\s*=\s*\([\w\.]+\)\s*([\w\.]+)', '\\1 = \\2', s)
            s = substitute('\(\([\w\.]+\) *([\w\.]+)\)', '(\\1)', s)

            s = substitute('\t', '    ', s)
            s = substitute('(?<!\w)Pattern\.', 're.', s)

            if buffer:
                start = 0
                index = 0
                buffer = []
                for m in re.compile('java_string').finditer(s):
                    buffer.append(s[start: m.start()])
                    buffer.append(string[index])
                    index += 1
                    start = m.end()

                buffer.append(s[start:])

                s = ''.join(buffer)

            arr.append(s)

    with open(python_file, 'w', encoding='utf8') as file:
        for s in arr:
            print(s, file=file)
#             print(s)


def formatPython(file):

    with open(file, 'r', encoding='utf8') as f:
        arr = f.readlines()
        s = arr[0]
        if s.endswith('\n'):
            s = s[:-1]
        print(s)

        m = re.compile('\S').search(s)

        if iscomment(s):
            root = Struct(s)
        else:
            if not m:
                indent = 0
            else:
                indent = m.start()
            root = Struct(s, indent)

        struct = root

        for i in range(1, len(arr)):
            s = arr[i]

            if s.endswith('\n'):
                s = s[:-1]
            print(s)

#             if s.strip().startswith('"The vocabulary file that the BERT model was trained on.'):
#                 print(s)
            struct = struct.encounter(s)

    root.validate()

    with open(file, 'w', encoding='utf8') as file:
        print(root, file=file)


def availableGPU():

    import pynvml  # @UnresolvedImport
    pynvml.nvmlInit()
    # 这里的0是GPU id
    maxFreeMemory = 0
    maxFreeMemoryID = 0
    for i in range(pynvml.nvmlDeviceGetCount()):
        print('the %dth GPU info:' % i)
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('used memory = ', meminfo.used / (1 << 20))
        print('free memory = ', meminfo.free / (1 << 20))
        print('total memory = ', meminfo.total / (1 << 20))
        if meminfo.free > maxFreeMemory:
            maxFreeMemoryID = i
            maxFreeMemory = meminfo.free

    print('GPU with the maximum Free Memory is %d, with Free Memory of %f MiB' % (maxFreeMemoryID, maxFreeMemory / (1 << 20)))
    return maxFreeMemoryID


def upper(ss):
    for i in range(len(ss)):
        ss[i] = ss[i].upper()
    return ss


def lower(ss):
    for i in range(len(ss)):
        ss[i] = ss[i].lower()
    return ss


def extend(arr, mask, maxlength, padding_type):
    if isinstance(arr, tuple):
        arr = [*arr]

    padding = [mask] * (maxlength - len(arr))
    if padding_type == 'tailing':
        arr.extend(padding)
    elif padding_type == 'leading':
        arr = padding + arr
    else:
        assert padding_type == 'general'
        for mask in padding:
            arr.insert(random.randrange(0, len(arr)), mask)

    return arr


# arr is a 3-dimension array
# def format_data(arr, mask_value=0, padding='general'):
# def format_data(arr, mask_value=0, padding='leading'):
def format_data(arr, mask_value=0, padding='tailing'):
    '''
    
    :param arr:
    :param mask_value:
    :param shuffle: randomly insert the padding mask into the sequence！ this is used for testing masking algorithms!
    '''

    try:
        maxWidth = max(len(x) for x in arr)
    except (TypeError, AttributeError) as _:
        return np.array(arr)

    try:
        maxHeight = max(max(len(word) for word in x) for x in arr)
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                arr[i][j] = extend(arr[i][j], mask_value, maxHeight, padding)
            arr[i] = extend(arr[i], [mask_value] * maxHeight, maxWidth, padding)
    except (TypeError, AttributeError) as _:

        # arr is a 2-dimension array
        for i in range(len(arr)):
            arr[i] = extend(arr[i], mask_value, maxWidth, padding)

    return np.array(arr)


def batch(function, instance, batch_size=32):
    res = []
    for i in range(0, len(instance), batch_size):
        res.append(function(instance[i:i + batch_size]))
    return np.concatenate(res, axis=0)


def invoke(cfunction, restype, argstype=None):
    cfunction.restype = restype

    if argstype is not None:
        cfunction.argstype = argstype
    return cfunction().contents


def counterDict(arr):
    dic = {}
    for x in arr:
        if x in dic:
            dic[x] += 1
        else:
            dic[x] = 1
    return dic


def split_sentence(s):
    length = len(s)
    minimum = length
    mid = length // 2

    ss = re.compile('[…，：；。！？℃、．（）]').split(s)

    index = 0

    length = 0
    for segment in ss:
#                     print(segment)
        index += len(segment.replace(' ', ''))
#                     print('index =', index)

        difference = abs(index - mid)
#                     print('difference =', difference)

        index += 1
        if difference < minimum:
            minimum = difference
            length += len(segment) + 1
        else:
            break
#                 if length < len(s):
    return s[:length], s[length:]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def plot2D(f, start, stop, step):
    from matplotlib import pyplot as plt

    low = lambda x:10000 if x > 10000 else -10000 if x < -10000 else x

    num = (stop - start) / step  # 计算点的个数
    x = np.linspace(start, stop, num)
    try:
        y = f(x)
    except TypeError:
        y = [f(e) for e in x]

    for i in range(len(y)):  # 再应用一个low函数以防止函数值过大导致显示错误（可选）#若函数无法直接应用到np数组上可以使用for来逐个应用
        y[i] = low(y[i])

#     fig = plt.figure(figsize=(12, 12))  # 建立一个对象并设置窗体的大小，使其为正方形，好看 #注意 可以建立多个对象，但plt指令只会对最后一个指定的对象进行操作（查看过源码了）
    fig = plt.figure()  # 建立一个对象并设置窗体的大小，使其为正方形，好看 #注意 可以建立多个对象，但plt指令只会对最后一个指定的对象进行操作（查看过源码了）

    plt.plot(x, y, label=f.__name__)  # 在当前的对象上进行操作

    plt.grid(True)  # 显示网格

    plt.axis("equal")  # 设置了x、y刻度长度一致#需要放在x、ylim指令前
    delta = (stop - start) / 16
    plt.xlim((start - delta, stop + delta))  # 显示的x的范围（不设置则由程序自动设置）

    ymin = min(y)
    ymax = max(y)

    delta = (ymax - ymin) / 16
    plt.ylim((ymin - delta, ymax + delta))

    plt.plot([2 * min(x), 2 * max(x)], [0, 0], label='x-axis')  # 用定义域最长距离的两倍作出x轴
    plt.plot([0, 0], [2 * min(y), 2 * max(y)], label='y-axis')  # 用值域最长距离的两倍作出y轴
    plt.legend()  # 显示旁注#注意：不会显示后来再定义的旁注

    plt.show(fig)  # 没有输入值默认展示所有对象 #注意：plt.show()之后再次使用plt.show()指令将不会展示任何对象，若想再次展示对象，可以对对象使用fig.show()
#     plt.show()  # 没有输入值默认展示所有对象 #注意：plt.show()之后再次使用plt.show()指令将不会展示任何对象，若想再次展示对象，可以对对象使用fig.show()


def plot2D_multiple(fs, start, stop, step):
    from matplotlib import pyplot as plt

    low = lambda x:10000 if x > 10000 else -10000 if x < -10000 else x

    num = (stop - start) / step  # 计算点的个数
    x = np.linspace(start, stop, num)
    y = [None] * len(fs)
    try:
        for i, f in enumerate(fs):
            y[i] = f(x)
    except TypeError:
        for i, f in enumerate(fs):
            y[i] = [f(e) for e in x]

    for _y in y:
        for i in range(len(_y)):  # 再应用一个low函数以防止函数值过大导致显示错误（可选）#若函数无法直接应用到np数组上可以使用for来逐个应用
            _y[i] = low(_y[i])

#     fig = plt.figure(figsize=(12, 12))  # 建立一个对象并设置窗体的大小，使其为正方形，好看 #注意 可以建立多个对象，但plt指令只会对最后一个指定的对象进行操作（查看过源码了）
    fig = plt.figure()  # 建立一个对象并设置窗体的大小，使其为正方形，好看 #注意 可以建立多个对象，但plt指令只会对最后一个指定的对象进行操作（查看过源码了）

    for _y, f in zip(y, fs):
        plt.plot(x, _y, label=f.__name__)  # 在当前的对象上进行操作

    plt.grid(True)  # 显示网格

    plt.axis("equal")  # 设置了x、y刻度长度一致#需要放在x、ylim指令前
    delta = (stop - start) / 16
    plt.xlim((start - delta, stop + delta))  # 显示的x的范围（不设置则由程序自动设置）

#     ymin = min(y)
#     ymax = max(y)
#     delta = (ymax - ymin) / 16
#     plt.ylim((ymin - delta, ymax + delta))

    plt.plot([2 * min(x), 2 * max(x)], [0, 0], label='x-axis')  # 用定义域最长距离的两倍作出x轴
#     plt.plot([0, 0], [2 * min(y), 2 * max(y)], label='y-axis')  # 用值域最长距离的两倍作出y轴
    plt.legend()  # 显示旁注#注意：不会显示后来再定义的旁注

    plt.show(fig)  # 没有输入值默认展示所有对象 #注意：plt.show()之后再次使用plt.show()指令将不会展示任何对象，若想再次展示对象，可以对对象使用fig.show()
#     plt.show()  # 没有输入值默认展示所有对象 #注意：plt.show()之后再次使用plt.show()指令将不会展示任何对象，若想再次展示对象，可以对对象使用fig.show()


def test_argmin():
    A = np.random.randint(10000, size=(1000, 1000, 1000))

    start = time.time()
    A.argmin(1)
    print('time cost =', time.time() - start)

    start = time.time()
    A.argmin(0)
    print('time cost =', time.time() - start)

    start = time.time()
    A.argmin(2)
    print('time cost =', time.time() - start)


def test_plot2D():
    m = 3

#     plot2D(lambda x : math.cos(m * x), -math.pi / m, math.pi / m, 0.01)
    def argular(x):
        k = int(math.floor(x * m / math.pi))
        return math.cos(m * x) * (-1 if k % 2 else 1) - 2 * k

    plot2D_multiple([argular, math.cos, lambda x : math.cos(m * x)], -math.pi, math.pi, 0.01)


def get_process_pid(tcp):
    if isinstance(tcp, int):
        tcp = str(tcp)
    res = os.popen('lsof -i tcp:%s' % tcp).readlines()
#     res = os.popen('lsof -i :%s' % tcp).readlines()
    for s in res[1:]:
        m = re.compile('gunicorn +(\d+) +').match(s)
#         pstree -ap|grep gunicorn
        if m :
            pid = m.group(1)
            print(s)
            yield pid


def get_tcp_from_url_root(url_root):
    m = re.search(":(\d+)/", url_root)
    assert m
    tcp = m.group(1)
    return tcp


def restart(tcp, sleep=60):
    print('tcp =', tcp)
    current_pid = str(os.getpid())
    parent_pid = str(os.getppid())

    print('current_pid =', current_pid)
    print('parent_pid =', parent_pid)

    pid = [*get_process_pid(tcp)]
    pid = set(pid)
    pid.remove(current_pid)
    pid.remove(parent_pid)
    pid = [*pid]
    pid.append(current_pid)

    for p in pid:
        print('kill -9 ' + p)
        os.system('kill -9 ' + p)
        time.sleep(sleep)


def connect(func):

    def _func(self, *args, **kwrags):
        with self:
            try:
                return func(self, *args, **kwrags)
            except Exception as e:
                print(e)
                traceback.print_exc()

    return _func

#
# def cursor(func):
#
#     def _func(self, *args, **kwrags):
#         self.cursor_stack.append(self.conn.cursor())
#         self.cursor = self.cursor_stack[-1]
#         res = None
#         try:
#             res = func(self, *args, **kwrags)
#         except Exception as e:
#             print(e)
#             traceback.print_exc()
#         self.cursor.close()
#         self.cursor_stack.pop()
#         if self.cursor_stack:
#             self.cursor = self.cursor_stack[-1]
#         else:
#             self.cursor = None
#         return res
#
#     return _func


def report_accuracy(func):

    def _func(self, *args, **kwrags):
        self.err = 0
        self.sum = 0

        result = func(self, *args, **kwrags)

        print('err =', self.err)
        print('sum =', self.sum)
        if self.sum:
            self.acc = (self.sum - self.err) / self.sum
            print('acc =', self.acc)
        else:
            self.acc = 0

        return result

    return _func


class Database:

    def create_database(self):
        cursor = self.cursor
        try:
            cursor.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(self.DB_NAME))
        except Exception as err:
            print("Failed creating database: {}".format(err))

    def __init__(self, creator):
        conf = configparser.ConfigParser()
        conf.read(workingDirectory + 'config.ini')

        try:
            self.pool = PooledDB(creator, mincached=5, blocking=True, **conf[creator.__name__])
#             self.cursor_stack = []
#             self.cursor = None
        except Exception as e:
            print(e)

    #def __enter__(self):
        #self.conn = self.pool.connection()

    def __exit__(self, *_):
        try:
#             print('shut down connector')
            self.conn.close()
        except Exception as e:
            print(e)

    @property
    def cursor(self):
        return self.conn.cursor()

    @property
    def wait_timeout(self):
        cursor = self.cursor
        cursor.execute("show global variables like 'wait_timeout'")
        for Variable_name, Value in cursor:
            assert Variable_name == 'wait_timeout'
            return Value

    @property
    def max_allowed_packet(self):
        cursor = self.cursor
        cursor.execute("show global variables like 'max_allowed_packet'")
        for Variable_name, Value in cursor:
            assert Variable_name == 'max_allowed_packet'
            return Value

    def commit(self):
        self.conn.commit()

    def select(self, sql):
        cursor = self.cursor
        cursor.execute(sql)
        yield from cursor._cursor

    def execute(self, sql, *args):
        self.cursor.execute(sql, *args)
        self.commit()

    @connect
    def show_create_table(self, table):
        for _, sql in self.select("show create table %s" % table):
            return sql

    @connect
    def show_tables(self):
        tables = [table for table, *_ in self.select("show tables")]
#         tables.sort()
        return tables

    @connect
    def show_create_table_oracle(self, table):
        for _, sql in self.select("select table_name, dbms_metadata.get_ddl('TABLE','%s') from dual,user_tables where table_name='%s'" % (table, table)):
            return sql

    @connect
    def desc_oracle(self, table):
        return [args for args in self.select("select column_name,data_type,nullable from all_tab_columns where owner='%s' and table_name='%s'" % (self.conn._con._kwargs['user'], table))]


def parseDateFormat(time):
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S")


def createNewFile(path):
    createNewPath(os.path.dirname(path))

    try:
        os.mknod(path)
    except :
        with open(path, 'w') as _:
            ...


def createNewPath(basedir):
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def config_ini(file, dic=None):
    if dic:
        with open(file, 'w') as file:
#             pickle.dump(dic)
            for k, v in dic.items():
                print('%s=%s' % (k, v), file=file)
    else:
        dic = {}
        for line in Text(file):
            m = re.compile("(.+?) *= *(\S+)").fullmatch(line)
            if m:
                key, value = m.groups()
                dic[key] = value

    return dic


def pickle_obj(file, obj=None):

    if obj is not None:
        with open(file, 'wb') as f:
            pickle.dump(obj, f)
    else:
        with open(file, 'rb') as f:
            obj = pickle.load(f)

    return obj


def get_size(obj, seen=None):
# From https://goshippo.com/blog/measure-real-size-any-python-object/
# Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


requests.adapters.DEFAULT_RETRIES = 5


def java(function, data={}, ip='localhost', evaluate=True):
    s = requests.session()
    s.keep_alive = False

#     r = requests.post("http://%s:8080/nlp/algorithm/%s" % (ip, function), data=data, headers={"content-type": "application/x-www-form-urlencoded;charset=utf8"})
    r = s.post("http://%s:8080/nlp/algorithm/%s" % (ip, function), data=data, headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf8", 'Connection':'close'})  # , timeout=30
    if evaluate:
        return json.loads(r.text)
    return r.text


def python(function, data, ip='localhost', evaluate=True):
    s = requests.session()
    s.keep_alive = False

#     r = requests.post("http://%s:8080/nlp/algorithm/%s" % (ip, function), data=data, headers={"content-type": "application/x-www-form-urlencoded;charset=utf8"})
    r = s.post("http://%s:8000/%s" % (ip, function), data=data, headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf8", 'Connection':'close'})  # , timeout=30
    if evaluate:
        return json.loads(r.text)
    return r.text


if __name__ == '__main__':

    result = java('ahocorasick', {'text' : '播放小晏晏的歌', 'service' : 'music'}, ip='192.168.2.39')
    print(result)
#     translateJava('../sequence/dependencyTreeReader.py')
#     translateJava('../sequence/compiler.py')
#     translateJava('../sequence/anomaly.py')
#     translateJava('../sequence/syntactic.py')
#     translateJava('../dialog/sentence.py')
#     translateJava('../dialog/converse.py')
#     translateJava('../dialog/qaCouplet.py')
#     translateJava('../classification/hierarchical/hierarchical.py')
#     translateJava('../classification/hierarchical/topicClassifier.py')
#     translateJava('../classification/hierarchical/paragraph.py')
#     translateJava('../util/utility_oracle.py')

#     test_argmin()
