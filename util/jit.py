'''
http://numba.pydata.org/numba-doc/latest/user/index.html
@author: Cosmos
'''
from _functools import reduce
from builtins import range
import math
import os
import re
import struct
import warnings

from numba.decorators import njit

import numpy as np
import tensorflow as tf
from util.utility import Text

# njit = njit(parallel=True, fastmath=True)
njit = njit(fastmath=True)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@njit
def errorMark(length, index):
    tag = ["  "] * length

    for i in index:
        tag[i] = "--"

    return tag


@njit
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
            l[i] = length(arr[i][j])

        maxLength = max(l)
        for i in range(len(arr)):
            res[i] += ' ' * (maxLength - l[i])

    return res


@njit
def length(value):
    length = 0
    for ch in value:
        if (ord(ch) & 0xff80) != 0:
            length += 2
        else:
            length += 1

    return length


@njit
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


@njit
def containNaN(matrix):
    if type(matrix) not in (np.ndarray, list) :
        return matrix != matrix
    for x in matrix:
        if containNaN(x):
            return True
    return False


@njit
def validateNaN(matrix):
    if len(matrix.shape) == 1:
        for i in range(len(matrix)):
            if matrix[i] != matrix[i]:
                matrix[i] = 0;

    if type(matrix) != np.ndarray:
        return
    for x in matrix:
        validateNaN(x)


@njit
def string2Index(char2id, s, maxlen=None):
    arr = [char2id.get(i, 0) for i in s]
    if maxlen:
        if len(arr) > maxlen:
            arr = arr[:maxlen]
        else :
            arr = arr + [-1] * (maxlen - len(arr))
    return arr


@njit
def readFolder(rootdir, sufix='.txt'):
    map = {}

    for name in os.listdir(rootdir):
        path = os.path.join(rootdir, name)

        if path.endswith(sufix):
            map[name[0:-len(sufix)]] = Text(path).collect()

    return map


#      the output variable reaches its peak when the input variable becomes 1/2 ln3;
def sinusoidalHyperbolicTangentln3(x) :
    return math.sqrt(math.sin(math.tanh(x) * math.pi))


#      the output variable reaches its peak when the input variable becomes T;
def sinusoidalHyperbolicTangent(x, T):
    return sinusoidalHyperbolicTangentln3(0.5 * math.log(3) * x / T);

# coding=utf-8
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d


def conv2d_batch(input, filter, strides, padding=None):
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


@njit
def initial_offset(x, y, w, s, i):
    if y.shape[i] > 1:
        l = x.shape[i] + (w.shape[i] - s[i]) * (y.shape[i] - 1)
        if y.shape[i] * w.shape[i] < l:
            l = y.shape[i] * w.shape[i]
        return w.shape[i] - (2 * w.shape[i] + l - (l + w.shape[i] - 1) // w.shape[i] * w.shape[i] + 1) // 2
    else:
        return -((x.shape[i] - w.shape[i]) // 2)


# stride=(1, 1)
@njit
def conv2d_same(x, w, bias, s, activate):
#     assert x.dtype == w.dtype
    yshape0 = (x.shape[0] + s[0] - 1) // s[0]
    yshape1 = (x.shape[1] + s[1] - 1) // s[1]
    y = np.zeros((yshape0, yshape1, w.shape[3]), dtype=x.dtype)
    d0 = initial_offset(x, y, w, s, 0)
    d1 = initial_offset(x, y, w, s, 1)
    for i in range(yshape0):
        for j in range(yshape1):
            _i = s[0] * i - d0
            _j = s[1] * j - d1

            for di in range(max(0, -_i), min(w.shape[0], x.shape[0] - _i)):
                for dj in range(max(0, -_j), min(w.shape[1], x.shape[1] - _j)):
                    y[i, j] += x[_i + di, _j + dj] @ w[di, dj]

            y[i, j] += bias
            y[i, j] = activate(y[i, j])
    return y


# stride=(1, 1)
@njit
def conv1d_same(x, w, bias, s, activate):
    yshape0 = (x.shape[0] + s - 1) // s
    y = np.zeros((yshape0, w.shape[2]), dtype=x.dtype)
    d0 = initial_offset(x, y, w, (s,), 0)
    for i in range(yshape0):
        _i = s * i - d0
        for di in range(max(0, -_i), min(w.shape[0], x.shape[0] - _i)):
            y[i] += x[_i + di] @ w[di]

        y[i] += bias
        y[i] = activate(y[i])
    return y


# stride=(1, 1)
@njit
def conv2d_valid(x, w, bias, s, activate):
    yshape0 = (x.shape[0] + s[0] - w.shape[0]) // s[0]
    yshape1 = (x.shape[1] + s[1] - w.shape[1]) // s[1]

    y = np.zeros((yshape0, yshape1, w.shape[3]), dtype=x.dtype)
    for i in range(yshape0):
        for j in range(yshape1):
            for di in range(w.shape[0]):
                for dj in range(w.shape[1]):
                    y[i, j] += x[s[0] * i + di, s[1] * j + dj] @ w[di, dj]
            y[i, j] += bias
            y[i, j] = activate(y[i, j])
    return y


@njit
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
@njit
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


@njit
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


def test_conv2d():
    in_height = 10
    in_width = 7
    in_channels = 1

    filter_height = 9
    filter_width = 7
    out_channels = 1
    strides_t = (5, 6)

    input_t = np.random.randint(10, size=(in_height, in_width, in_channels))
    filter_t = np.random.randint(10, size=(filter_height, filter_width, in_channels, out_channels))

    print('input_t:')
    print(input_t)
    print('filter_t:')
    print(filter_t)

    print('numpy conv2d:')
    res = conv2d_valid(input_t, filter_t, None, strides_t, None)
#     res = max_pooling(input_t, padding=padding)
    print(res.shape)
    print(res)

    _res = conv2d_same(input_t, filter_t, None, strides_t, None)
    print(_res.shape)
    print(_res)

    print('tensorflow conv2d')

    a = tf.Variable(input_t.reshape((1, in_height, in_width, in_channels)), dtype=tf.float32)
    b = tf.Variable(filter_t, dtype=tf.float32)
    op = tf.nn.conv2d(a, b, strides=(1, strides_t[0], strides_t[1], 1), padding='VALID')
    _op = tf.nn.conv2d(a, b, strides=(1, strides_t[0], strides_t[1], 1), padding='SAME')
#     op = tf.nn.max_pool(a, [1, 2, 2, 1], strides=(1, 2, 2, 1), padding=padding)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        res_tf = sess.run(op)

        res_tf = res_tf.reshape(res_tf.shape[1:])
        print(res_tf.shape)
        print(res_tf)
        assert res_tf.shape == res.shape
        print('res_tf - res')
        print(res_tf - res)

        assert (res_tf == res).all()

        _res_tf = sess.run(_op)

        _res_tf = _res_tf.reshape(_res_tf.shape[1:])
        print(_res_tf.shape)
        print(_res_tf)
        assert _res_tf.shape == _res.shape
        print('_res_tf - _res')
        print(_res_tf - _res)

        assert (_res_tf == _res).all()


def test_conv1d():
    in_width = 7
    in_channels = 2

    filter_width = 3
    out_channels = 3
    strides_t = 3

    input_t = np.random.randint(10, size=(in_width, in_channels))
    filter_t = np.random.randint(10, size=(filter_width, in_channels, out_channels))

    print('input_t:')
    print(input_t)
    print('filter_t:')
    print(filter_t)

    print('numpy conv2d:')
    res = conv1d_valid(input_t, filter_t, None, strides_t, None)
#     res = max_pooling(input_t, padding=padding)
    print(res.shape)
    print(res)

    _res = conv1d_same(input_t, filter_t, None, strides_t, None)
    print(_res.shape)
    print(_res)

    print('tensorflow conv2d')

    a = tf.Variable(input_t.reshape((1, in_width, in_channels)), dtype=tf.float32)
    b = tf.Variable(filter_t, dtype=tf.float32)
    op = tf.nn.conv1d(a, b, stride=strides_t, padding='VALID')
    _op = tf.nn.conv1d(a, b, stride=strides_t, padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        res_tf = sess.run(op)

        res_tf = res_tf.reshape(res_tf.shape[1:])
        print(res_tf.shape)
        print(res_tf)
        assert res_tf.shape == res.shape
        print('res_tf - res')
        print(res_tf - res)

        assert (res_tf == res).all()

        _res_tf = sess.run(_op)

        _res_tf = _res_tf.reshape(_res_tf.shape[1:])
        print(_res_tf.shape)
        print(_res_tf)
        assert _res_tf.shape == _res.shape
        print('_res_tf - _res')
        print(_res_tf - _res)

        assert (_res_tf == _res).all()
# coding=utf-8
# https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d


def test_conv2d_batch():
    batch = 1
    in_height = 5
    in_width = 5
    in_channels = 3

    filter_height = 2
    filter_width = 2
    out_channels = 2

    input_t = np.random.randint(10, size=(batch, in_height, in_width, in_channels))
    filter_t = np.random.randint(10, size=(filter_height, filter_width, in_channels, out_channels))
    strides_t = [1, 1, 1, 1]
    print('numpy conv2d:')
    res = conv2d_batch(input_t, filter_t, strides_t)
    print(res)
    print(res.shape)
    print('tensorflow conv2d')

    a = tf.Variable(input_t, dtype=tf.float32)
    b = tf.Variable(filter_t, dtype=tf.float32)
    op = tf.nn.conv2d(a, b, strides=strides_t, padding='VALID')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _res = sess.run(op)
        assert _res.shape == res.shape
        assert (_res == res).all()

        print(_res)
        print(_res.shape)


def separable_conv2d(
    input,
    depth_filter,
    point_fileter,
    strides,
    rate=[1, 1],
    padding=None
):
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


def test_separable_conv2d():
    # input_t = np.ones([3,5,5,3])
    # depth_filter_t = np.ones([2,2,3,2])
    # point_filter_t = np.ones([1,1,6,2])
    input_t = np.random.randint(10, size=(3, 5, 5, 3))
    depth_filter_t = np.random.randint(10, size=(2, 2, 3, 2))
    point_filter_t = np.random.randint(10, size=(1, 1, 6, 2))
    strides_t = [1, 1, 1, 1]
    print('numpy separable_conv2d:')
    res = separable_conv2d(input_t, depth_filter_t, point_filter_t, strides_t)
    print(res)
    print(res.shape)
    print('tensorflow separable_conv2d')
    a = tf.Variable(input_t, dtype=tf.float32)
    b = tf.Variable(depth_filter_t, dtype=tf.float32)
    c = tf.Variable(point_filter_t, dtype=tf.float32)
    op = tf.nn.separable_conv2d(a, b, c,
        strides=strides_t,
        padding='VALID')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(op)
        print(res)
        print(res.shape)

# https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d


def depthwise_conv2d(
    input,
    filter,
    strides,
    rate=[1, 1],
    padding=None
):
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


def test_depthwise_conv2d():
    input_t = np.random.randint(10, size=(3, 5, 5, 3))
    filter_t = np.random.randint(10, size=(2, 2, 3, 2))
    strides_t = [1, 1, 1, 1]
    print('numpy depthwise_conv2d:')
    res = depthwise_conv2d(input_t, filter_t, strides_t)
    print(res)
    print(res.shape)
    print('tensorflow depthwise_conv2d')
    a = tf.Variable(input_t, dtype=tf.float32)
    b = tf.Variable(filter_t, dtype=tf.float32)
    op = tf.nn.depthwise_conv2d(a, b,
        strides=strides_t,
        padding='VALID')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(op)
        print(res)
        print(res.shape)


def relu_inplace(feature_map):
    # Preparing the output of the ReLU activation function.
    for map_num in range(feature_map.shape[-1]):
        for r in range(feature_map.shape[0]):
            for c in range(feature_map.shape[1]):
                feature_map[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])


@njit
def relu(x):
    return x * (x > 0)


@njit
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0)


@njit
def linear(x):
    return x


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit
def hard_sigmoid(x):
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    x = np.full_like(x, 0.2) * x + np.full_like(x, 0.5)
    x = x * (x >= zero) - one
    return x * (x <= zero) + one


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
    return re.compile('[a-zA-Zａ-ｚＡ-Ｚ]').match(c)


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

# from numba import jitclass  # import the decorator


# @jitclass([('Wxu', np.int), ('Wxr', np.float[:]), ])
class GRU:

    @njit
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

    @njit
    def __init__(self, Wxu, Wxr, Wxh, Whu, Whr, Whh, bu, br, bh):
        self.Wxu = Wxu
        self.Wxr = Wxr
        self.Wxh = Wxh

        self.Whu = Whu
        self.Whr = Whr
        self.Whh = Whh

        self.bu = bu
        self.br = br
        self.bh = bh

        self.sigmoid = hard_sigmoid
        self.tanh = np.tanh

    @njit
    def activate(self, x, h):
        if h is None:
            u = np.dot(x, self.Wxu)
            if self.bu is not None:
                u += self.bu
            u = self.sigmoid(u)

            gh = np.dot(x, self.Wxh)
            if self.bh is not None:
                gh += self.bh
            gh = self.tanh(gh)

            return (1 - u) * gh
        else:
            r = np.dot(x, self.Wxr) + np.dot(h, self.Whr)
            if self.br is not None:
                r += self.br
            r = self.sigmoid(r)

            u = np.dot(x, self.Wxu) + np.dot(h, self.Whu)
            if self.bu is not None:
                u += self.bu
            u = self.sigmoid(u)

            gh = np.dot(x, self.Wxh) + np.dot(r * h, self.Whh)
            if self.bh is not None:
                gh += self.bh
            gh = self.tanh(gh)

            return (1 - u) * gh + u * h

#     // forward pass
    @njit
    def propogate(self, seq):
        h = None

        for xt in seq:
            h = self.activate(xt, h)

        return h

    @njit
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
    @njit
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

    @njit
    def weights(self):
        return np.concatenate([self.Wxu, self.Wxr, self.Wxh], axis=1), np.concatenate([self.Whu, self.Whr, self.Whh], axis=1), np.concatenate([self.bu, self.br, self.bh], axis=0)


class LSTM:

    @njit
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

    def __init__(self, Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo):
        self.Wxi = Wxi
        self.Wxf = Wxf
        self.Wxc = Wxc
        self.Wxo = Wxo

        self.Whi = Whi
        self.Whf = Whf
        self.Whc = Whc
        self.Who = Who

        self.bi = bi
        self.bf = bf
        self.bc = bc
        self.bo = bo

        self.sigmoid = hard_sigmoid
        self.tanh = np.tanh

    @njit
    def activate(self, x, h, c):
        if h is None:
            i = self.sigmoid(np.dot(x, self.Wxi) + self.bi)
            c = i * self.tanh(np.dot(x, self.Wxc) + self.bc)
            o = self.sigmoid(np.dot(x, self.Wxo) + self.bo)
        else:
            i = self.sigmoid(np.dot(x, self.Wxi) + np.dot(h, self.Whi) + self.bi)
            f = self.sigmoid(np.dot(x, self.Wxf) + np.dot(h, self.Whf) + self.bf)
            c = f * c + i * self.tanh(np.dot(x, self.Wxc) + np.dot(h, self.Whc) + self.bc)
            o = self.sigmoid(np.dot(x, self.Wxo) + np.dot(h, self.Who) + self.bo)
        return o * self.tanh(c), c

#     // forward pass
    @njit
    def propogate(self, x, return_sequences=False):
        h = None
        c = None
        if return_sequences:
            arr = np.zeros(x.shape)  # The desired data-type for the array, Default is numpy.float64.

        for i in range(len(x)):
            h, c = self.activate(x[i], h, c)
            if return_sequences:
                arr[i] = h

        if return_sequences:
            return arr

        return h

    @njit
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
    @njit
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

    @njit
    def weights(self):
        return np.concatenate([self.Wxi, self.Wxf, self.Wxc, self.Wxo], axis=1), np.concatenate([self.Whi, self.Whf, self.Whc, self.Who], axis=1), np.concatenate([self.bi, self.bf, self.bc, self.bo], axis=0)


@njit
def lstm_activate0(Wxi, Wxc, Wxo, bi, bc, bo, x):
    i = hard_sigmoid(x @ Wxi + bi)
    c = i * np.tanh(x @ Wxc + bc)
    o = hard_sigmoid(x @ Wxo + bo)
    return o * np.tanh(c), c


@njit
def lstm_activate(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, x, h, c):
    i = hard_sigmoid(x @ Wxi + h @ Whi + bi)
    f = hard_sigmoid(x @ Wxf + h @ Whf + bf)
    c = f * c + i * np.tanh(x @ Wxc + h @ Whc + bc)
    o = hard_sigmoid(x @ Wxo + h @ Who + bo)
    return o * np.tanh(c), c


@njit
def lstm_propogate(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, x):
    h, c = lstm_activate0(Wxi, Wxc, Wxo, bi, bc, bo, x[0])

    for i in range(1, len(x)):
        h, c = lstm_activate(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, x[i], h, c)

    return h


@njit
def lstm_propogate_return_sequences(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, X):
    arr = np.zeros(X.shape, X.dtype)  # The desired data-type for the array, Default is numpy.float64.

    h, c = lstm_activate0(Wxi, Wxc, Wxo, bi, bc, bo, X[0])
    arr[0] = h

    for k in range(1, len(X)):
        h, c = lstm_activate(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, X[k], h, c)
        arr[k] = h

    return arr


class CRF:

    @njit
    def __eq__(self, autre):
        return (self.bias == autre.bias).all()\
            and (self.chain_kernel == autre.chain_kernel).all()\
            and (self.kernel == autre.kernel).all()\
            and (self.left_boundary == autre.left_boundary).all()\
            and (self.right_boundary == autre.right_boundary).all()

    def __init__(self, kernel, chain_kernel, bias, left_boundary, right_boundary):
        self.bias = bias

        self.chain_kernel = chain_kernel

        self.kernel = kernel

        self.left_boundary = left_boundary

        self.right_boundary = right_boundary

        self.activation = linear

    @njit
    def viterbi_one_hot(self, X):
        return np.eye(len(self.bias), dtype=int)[self.viterbi_decoding(X)]

    @njit
    def viterbi_decoding(self, X):
        input_energy = self.activation(np.matmul(X, self.kernel) + self.bias)

        input_energy[0] += self.left_boundary
        input_energy[-1] += self.right_boundary

        min_energy = np.zeros_like(input_energy[0])

        argmin_tables = np.zeros(input_energy.shape, int)
        for t in range(len(input_energy)):
            energy = self.chain_kernel + np.expand_dims(input_energy[t] + min_energy, 1)
            argmin_tables[t] = np.argmin(energy, 0)
            min_energy = np.min(energy, 0)

        best_idx = np.argmin(min_energy)
#         best_idx = argmin_tables[-1, 0]
        best_paths = np.zeros((len(argmin_tables),), int)
        for i in range(-1, -len(argmin_tables) - 1, -1):
            best_idx = argmin_tables[i][best_idx]
            best_paths[i] = best_idx

        return best_paths

    @njit
    def write(self, file):
        writeArray(file, self.kernel)
        writeArray(file, self.chain_kernel)
        writeArray(file, self.bias)
        writeArray(file, self.left_boundary)
        writeArray(file, self.right_boundary)

    @staticmethod
    @njit
    def read(file):
        kernel = readArray(file, 2)
        chain_kernel = readArray(file, 2)
        bias = readArray(file, 1)
        left_boundary = readArray(file, 1)
        right_boundary = readArray(file, 1)

        return CRF(kernel, chain_kernel, bias, left_boundary, right_boundary)

    @njit
    def weights(self):
        return self.kernel, self.chain_kernel, self.bias, self.left_boundary, self.right_boundary


@njit
def viterbi_decoding(kernel, chain_kernel, bias, left_boundary, right_boundary, X):
    input_energy = X @ kernel + bias

    input_energy[0] += left_boundary
    input_energy[-1] += right_boundary

    min_energy = np.zeros_like(input_energy[0])

    argmin_tables = np.zeros(input_energy.shape, np.int32)

#     chain_kernel = chain_kernel.T

    for t in range(len(input_energy)):
        energy = np.zeros_like(chain_kernel)
        column = input_energy[t] + min_energy
        for i in range(energy.shape[0]):
            for j in range(energy.shape[1]):
                energy[j][i] = column[i]

        energy += chain_kernel

        for i in range(energy.shape[0]):
            j = energy[i].argmin()
            argmin_tables[t][i] = j
            min_energy[i] = energy[i, j]

    best_idx = min_energy.argmin()
    best_paths = np.zeros(len(argmin_tables), np.int32)
    for i in range(-1, -len(argmin_tables) - 1, -1):
        best_idx = argmin_tables[i][best_idx]
        best_paths[i] = best_idx

    return best_paths


@njit
def viterbi_one_hot(kernel, chain_kernel, bias, left_boundary, right_boundary, X):
    return np.eye(len(bias), dtype=np.int32)[viterbi_decoding(kernel, chain_kernel, bias, left_boundary, right_boundary, X)]


class Embedding:

    @njit
    def __eq__(self, autre):
        return self.char2id == autre.char2id and (self.wEmbedding == autre.wEmbedding).all()

    @njit
    def __init__(self, char2id, wEmbedding):
        self.char2id = char2id
        self.wEmbedding = wEmbedding

    @njit
    def call(self, word):
        length = len(word)
        charEmbedding = np.zeros([length, self.wEmbedding.shape[1]])

        for j in range(length):
            charEmbedding[j] = self.wEmbedding[self.char2id.get(word[j], 0)]

        return charEmbedding;

    @njit
    def write(self, file):
        writeCharDict(file, self.char2id)
        writeArray(file, self.wEmbedding)

    @staticmethod
    @njit
    def read(file):
        char2id = readCharDict(file)
        wEmbedding = readArray(file, 2)

        return Embedding(char2id, wEmbedding)


class Dense:

    @njit
    def __eq__(self, autre):
        return (self.wDense == autre.wDense).all() and (self.bDense == autre.bDense).all()

    @njit
    def __init__(self, wDense, bDense):
        self.wDense = wDense
        self.bDense = bDense

    @njit
    def call(self, x):
        return np.matmul(x, self.wDense) + self.bDense

    @njit
    def write(self, file):
        writeArray(file, self.wDense)
        writeArray(file, self.bDense)

    @staticmethod
    @njit
    def read(file):
        wDense = readArray(file, 2)
        bDense = readArray(file, 1)

        return Dense(wDense, bDense)


class Conv2d:

    @njit
    def __eq__(self, autre):
        return (self.wCNN == autre.wCNN).all() and (self.bCNN == autre.bCNN).all()

    @njit
    def __init__(self, wCNN, bCNN):
        self.wCNN = wCNN
        self.bCNN = bCNN
        self.paddingSame = False

    @njit
    def call(self, x):
        return conv2d_same(x, self.wCNN, self.bCNN, (1, 1), relu) if self.paddingSame else conv2d_valid(x, self.wCNN, self.bCNN, (1, 1), relu)

    @njit
    def write(self, file):
        writeArray(file, self.wCNN)
        writeArray(file, self.bCNN)

    @staticmethod
    @njit
    def read(file):
        wDense = readArray(file, 3)
        bDense = readArray(file, 1)

        return Conv2d(wDense, bDense)


class Conv1d:

    @njit
    def __eq__(self, autre):
        return (self.wCNN == autre.wCNN).all() and (self.bCNN == autre.bCNN).all()

    @njit
    def __init__(self, wCNN, bCNN):
        self.wCNN = wCNN
        self.bCNN = bCNN
        self.paddingSame = False

    @njit
    def call(self, x):
        return conv1d_same(x, self.wCNN, self.bCNN, 1, relu) if self.paddingSame else conv1d_valid(x, self.wCNN, self.bCNN, 1, relu)

    @njit
    def write(self, file):
        writeArray(file, self.wCNN)
        writeArray(file, self.bCNN)

    @staticmethod
    @njit
    def read(file):
        wDense = readArray(file, 3)
        bDense = readArray(file, 1)

        return Conv1d(wDense, bDense)


@njit
def writeArray(file, array):
    if len(array.shape) == 1:
        file.write(struct.pack('>i' , *array.shape))
        file.write(struct.pack('>' + str(array.size) + 'd', *array))
    else:
        file.write(struct.pack('>' + str(len(array.shape)) + 'i' , *array.shape))
        file.write(struct.pack('>' + str(array.size) + 'd', *array.reshape(-1)))


@njit
def readArray(file, dimension):
    if dimension > 1:
        fmt = '>' + str(dimension) + 'i'

        dimension = struct.unpack(fmt, file.read(struct.calcsize(fmt)))

        fmt = '>' + str(reduce(lambda x, y: x * y, dimension)) + 'd'
        arr = struct.unpack(fmt , file.read(struct.calcsize(fmt)))
        return np.array(arr).reshape(dimension)
    else:
        fmt = '>i'

        dimension = struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]

        fmt = '>' + str(dimension) + 'd'
        arr = struct.unpack(fmt , file.read(struct.calcsize(fmt)))
        return np.array(arr)


@njit
def writeCharDict(file, char2id):
    file.write(struct.pack('>i' , len(char2id)))
    for key, value in char2id.items():
        file.write(struct.pack('>H', ord(key)))
        file.write(struct.pack('>i', value))


@njit
def readCharDict(file):
    char2id = {}
    length = struct.unpack('>i' , file.read(struct.calcsize('i')))[0]

    for _ in range(length):
        fmt = '>H'

        c = struct.unpack(fmt, file.read(struct.calcsize(fmt)))[0]

        value = struct.unpack('>i', file.read(struct.calcsize('i')))[0]
        char2id[chr(c)] = value
    return char2id


@njit
def writeWordDict(file, word2id):
    file.write(struct.pack('>i' , len(word2id)))
    for key, value in word2id.items():
        file.write(struct.pack('>i', len(key)))
        file.write(struct.pack('>' + str(len(key)) + 'H', *[ord(x) for x in key]))
        file.write(struct.pack('>i', value))


@njit
def readWordDict(file):
    word2id = {}
    length = struct.unpack('>i' , file.read(struct.calcsize('i')))[0]

    for _ in range(length):
        l = struct.unpack('>i', file.read(struct.calcsize('i')))[0]
        fmt = '>' + str(l) + 'H'

        s = ''
        for c in struct.unpack(fmt, file.read(struct.calcsize(fmt))):
            s += chr(c)

        value = struct.unpack('>i', file.read(struct.calcsize('i')))[0]
        word2id[s] = value
    return word2id


@njit
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


@njit
def convertToSegment(predict_text, argmax):

    arr = []
    sstr = ''
    for i in range(len(predict_text)):
        if predict_text[i] != ' ':
            sstr = sstr + predict_text[i]
        if argmax[i] & 1 and len(sstr):
            arr.append(sstr)
            sstr = ''

    if len(sstr):
        arr.append(sstr)
    return arr


@njit
def string2id(char2id, s):
    arr = [char2id.get(i, 0) for i in s]
    return arr


@njit
def cws(wEmbedding, Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, wCNN0, bCNN0, wCNN1, bCNN1, kernel, chain_kernel, bias, left_boundary, right_boundary, predict_text, x):
    lEmbedding = np.zeros(shape=(len(x), wEmbedding.shape[1]), dtype=wEmbedding.dtype)
#     lEmbedding = wEmbedding[x], indexing with array of integers is not supported!
    for i in range(len(x)):
        lEmbedding[i] = wEmbedding[x[i]]

    lLSTM = lstm_propogate_return_sequences(Wxi, Wxf, Wxc, Wxo, Whi, Whf, Whc, Who, bi, bf, bc, bo, lEmbedding)

    lCNN = conv1d_same(lEmbedding, wCNN0, bCNN0, 1, relu)
    lCNN = conv1d_same(lCNN, wCNN1, bCNN1, 1, relu)

    lConcatenate = np.concatenate((lLSTM, lCNN), axis=1)

    lCRF = viterbi_decoding(kernel, chain_kernel, bias, left_boundary, right_boundary, lConcatenate)
    return convertToSegment(predict_text, lCRF)


if __name__ == '__main__':
    wEmbedding = np.random.randint(10, size=(6000, 64)).astype(np.float32)
    x = [0, 1 , 2 , 3]
    print(wEmbedding.shape)
    print(indexing(wEmbedding, x).shape)
