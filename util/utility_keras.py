'''

@author: Cosmos
'''
from builtins import range
import os
import warnings
import random
from keras.utils.data_utils import Sequence
import keras
from util import utility
from assertpy.assertpy import assert_that
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from keras.constraints import UnitNorm
from keras import initializers, regularizers, constraints

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf


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
    res = utility.conv2d_valid(input_t, filter_t, None, strides_t, None)
#     res = max_pooling(input_t, padding=padding)
    print(res.shape)
    print(res)

    _res = utility.conv2d_same(input_t, filter_t, None, strides_t, None)
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
    res = utility.conv1d_valid(input_t, filter_t, None, strides_t, None)
#     res = max_pooling(input_t, padding=padding)
    print(res.shape)
    print(res)

    _res = utility.conv1d_same(input_t, filter_t, None, strides_t, None)
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
    res = utility.conv2d_batch(input_t, filter_t, strides_t)
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


def test_separable_conv2d():
    # input_t = np.ones([3,5,5,3])
    # depth_filter_t = np.ones([2,2,3,2])
    # point_filter_t = np.ones([1,1,6,2])
    input_t = np.random.randint(10, size=(3, 5, 5, 3))
    depth_filter_t = np.random.randint(10, size=(2, 2, 3, 2))
    point_filter_t = np.random.randint(10, size=(1, 1, 6, 2))
    strides_t = [1, 1, 1, 1]
    print('numpy separable_conv2d:')
    res = utility.separable_conv2d(input_t, depth_filter_t, point_filter_t, strides_t)
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


def test_depthwise_conv2d():
    input_t = np.random.randint(10, size=(3, 5, 5, 3))
    filter_t = np.random.randint(10, size=(2, 2, 3, 2))
    strides_t = [1, 1, 1, 1]
    print('numpy depthwise_conv2d:')
    res = utility.depthwise_conv2d(input_t, filter_t, strides_t)
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


class Unmask(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Unmask, self).__init__(**kwargs)
        self.supports_masking = False

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, *_):
        return None

    def call(self, inputs):
        return inputs


class Generator(Sequence):

    def __init__(self, trainingArr, x_name='x', y_name='y', batch_size=32, epoch=1, shuffle=False, allocate=None, format_data=None):
        self.batch_size = batch_size
        self.trainingDict = {}

        for s in trainingArr:
            length = len(getattr(s, x_name))

            if length not in self.trainingDict:
                self.trainingDict[length] = []

            self.trainingDict[length].append(s)

        self.allocate = allocate

        self.x_name = x_name
        self.y_name = y_name
        self.format_data = format_data
        self.epoch = epoch
        self.shuffle = shuffle

    def __getitem__(self, index):
        return self.arr[index]

    def __len__(self):
        self.arr = [item for item in iter(self)]

        if self.shuffle:
            arr = []
            for _ in range(self.epoch):
                random.shuffle(self.arr)
                arr += self.arr
            self.arr = arr
        else:
            if self.epoch > 1:
                self.arr *= self.epoch

        return len(self.arr)

    def on_epoch_end(self):
        print('\none epoch has ended!')

    def original_list(self):
        for length in self.trainingDict:
            for sample in self.trainingDict[length]:
                yield sample

    def make_numpy(self, batch):
        x_sample = [getattr(s, self.x_name) for s in batch]
        if self.format_data:
            x_sample = self.format_data(x_sample)

        return np.array(x_sample), np.array([getattr(s, self.y_name) for s in batch])

    def __iter__(self):
        for length in self.trainingDict:
            samples = self.trainingDict[length]
            batch_size = self.batch_size
#             this line is used to prevent out of memory exception when training with tensorflow-gpu!
            if self.allocate:
                for size in self.allocate:
                    if length >= size:
                        batch_size //= 2

            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                yield self.make_numpy(batch)


class SequenceGenerator(Sequence):

    def __init__(self, trainingArr, x_name='x', y_name=None, batch_size=32, epoch=1, shuffle=False, allocate=None, format_x=None, format_y=None):
        self.batch_size = batch_size
        if isinstance(x_name, list):
            self.original_list = sorted(trainingArr, key=lambda s: len(getattr(s, x_name[0])))
        else:
            self.original_list = sorted(trainingArr, key=lambda s: len(getattr(s, x_name)))

        self.allocate = allocate

        self.x_name = x_name
        self.y_name = y_name
        self.format_x = format_x
        self.format_y = format_y if y_name else None
        self.epoch = epoch if y_name else 1
        self.shuffle = shuffle if y_name else False

    def __getitem__(self, index):
        return self.arr[index]

    def batch(self):
        return (self.original_list[i:i + self.batch_size] for i in range(0, len(self.original_list), self.batch_size))

    def __len__(self):
        if hasattr(self, 'arr'):
            return len(self.arr)

        self.arr = []

        for batch in self.batch():
            assert batch is not None
            self.arr.append(self.make_numpy(batch))

        if self.shuffle:
            arr = []
            for _ in range(self.epoch):
                random.shuffle(self.arr)
                arr += self.arr
            self.arr = arr
        else:
            if self.epoch > 1:
                self.arr *= self.epoch

        return len(self.arr)

    def on_epoch_end(self):
        print('\none epoch has ended!')

    def make_numpy(self, batch):
        assert batch is not None

        if isinstance(self.x_name, list):
            x_sample = map(lambda x_name: [getattr(s, x_name) for s in batch], self.x_name)
            if self.format_x:
                x_sample = map(self.format_x, x_sample)
            else:
                x_sample = map(np.array, x_sample)
            x_sample = [*x_sample]
        else:
            x_sample = [getattr(s, self.x_name) for s in batch]

            if self.format_x:
                x_sample = self.format_x(x_sample)
            else:
                x_sample = np.array(x_sample)

        if self.y_name:
            y_sample = [getattr(s, self.y_name) for s in batch]
            if self.format_y:
                y_sample = self.format_x(y_sample)
            else:
                y_sample = np.array(y_sample)
            return x_sample, y_sample

        return x_sample


def test_FIFOQueue():
    input_data = [[[3.5, 3.5], [2.5, 3.5], [1.5, 3.5], [3.4, 3.5], [2.4, 3.5], [1.4, 3.5]], [[(11,)], [(22,)], [(33,)], [(10,)], [(20,)], [(30,)]], [['string-111.'], ['string-222.'], ['string-333.'], ['string-11.'], ['string-22.'], ['string-33.']]]
    length = np.array([len(arr) for arr in input_data])

    assert_that((length == length.min()).all()).is_true()

    q = tf.FIFOQueue(length[0], dtypes=[tf.float32, tf.int32, tf.string], shapes=[(2,), (1, 1), (1,)])
    init = q.enqueue_many(input_data)

    dequeue_many = q.dequeue_many(3)
    dequeue = q.dequeue()
    with tf.Session() as sess:
        init.run()
        print('1：', sess.run(dequeue_many))
        print('2：', sess.run(dequeue))
        print('3：', sess.run(dequeue))
        print('4：', sess.run(dequeue))
        sess.run(q.close(cancel_pending_enqueues=True))
        print(sess.run(q.is_closed()))


class TensorArray:

    def __init__(self, dtype, size):
        self.array = tensor_array_ops.TensorArray(dtype=dtype, size=size)

    def __getitem__(self, index):
        return self.array.read(index)

    def __setitem__(self, index, value):
        self.array = self.array.write(index, value)

    def stack(self):
        return self.array.stack()


# perform y = func(x)
def for_loop(size, func, *args, **kwargs):

    def transform(x, stack=True, transpose=True):
        if stack:
            x = x.stack()

        if transpose:
            shape = [*range(len(x.shape))]
            shape[0], shape[1] = shape[1], shape[0]
            return tf.transpose(x, shape)
        else:
            return x

    def make_tuple(x):
        if isinstance(x, (tuple, list)):
            return x
        return (x,)

    if 'reverse' in kwargs and kwargs['reverse']:
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = -1
        stop = start - size
        _, *args = control_flow_ops.while_loop(lambda i, *_: i > stop ,
                                               lambda i, *args:(i - 1, *make_tuple(func(i, *args))),
                                               loop_vars=(start, *args))
    else:
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0
        stop = start + size

        _, *args = control_flow_ops.while_loop(lambda i, *_: i < stop,
                                               lambda i, *args:(i + 1, *make_tuple(func(i, *args))),
                                               loop_vars=(start, *args))
    transpose = 'transpose' in kwargs and kwargs['transpose']
    stack = 'stack' not in kwargs or kwargs['stack']

    args = [transform(arg, stack, transpose) if isinstance(arg, tensor_array_ops.TensorArray) else arg for arg in args]
    return args[0] if len(args) == 1 else args


class build_sublayer(type):

    def __new__(self, name, bases, attrs, *args, **kwargs):

        def build_sublayer(self, sublayer, input_shape):
            sublayer.build(input_shape)
            self._trainable_weights.extend(sublayer._trainable_weights)
            self._non_trainable_weights.extend(sublayer._non_trainable_weights)
            self.add_loss(sublayer._losses)
            self.output_shape = sublayer.compute_output_shape(input_shape)
            return self.output_shape

        def compute_output_shape(self, _):
            return self.output_shape

        attrs['output_shape'] = ()
        attrs['build_sublayer'] = build_sublayer
        attrs['compute_output_shape'] = compute_output_shape

        return type.__new__(self, name, bases, attrs, *args, **kwargs)

#     def __init__(self, name, bases, attrs, *args, **kwargs):
#         print('def __init__(self):')
#         super(build_sublayer, self).__init__(name, bases, attrs, *args, **kwargs)

#     Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED


keras.backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False)))
# the following initialization should occur before model initialization!
# to avoid the error: tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value crf_1/bias

keras.backend.get_session().run(tf.global_variables_initializer())


class Model:

    @property
    def dimension(self):
        return len(self.char2id) + 2

    def string2id(self, s):
        return [self.char2id.get(i, 1) for i in s]

    def expand_vocabulary(self, charSet):
        print('new characters found!')
        print(charSet)

        index = self.dimension
        self.char2id.update({word : i + index for i, word in enumerate(charSet)})
        weights = self.model.get_weights()
        shape = weights[0].shape
        dimensionAdded = self.dimension - shape[0]
        assert dimensionAdded > 0

        weights[0] = np.append(weights[0], np.zeros((dimensionAdded, shape[1])), 0)

        del self.model
        self.create()
        self.model.set_weights(weights)

    def plot(self):
        from keras.utils.vis_utils import plot_model
        plot_model(self.model, to_file=self.modelFile + '.png', show_shapes=True)


import keras.backend as K


class AMSoftmax(keras.layers.Layer):

    def __init__(self, output_dim,
                 kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='FAN_AVG', distribution="normal"),
                 kernel_constraint=UnitNorm(),
                 kernel_regularizer=None,
                 classification=False,
                 sparse=False,
                 **kwargs):
        super(AMSoftmax, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.m = 0.35
        self.s = 30
        self.classification = classification
        self.sparse = sparse

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.output_dim),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.built = True

    def compute_output_shape(self, *_):
        return (None, self.output_dim)

    def accuracy(self, label_batch, cos_theta):
        return keras.metrics.categorical_accuracy(label_batch, cos_theta)

    def loss(self, y_true, y_pred):
        if self.sparse:
            y_true = K.expand_dims(y_true[:, 0], 1)
            y_true = K.cast(y_true, 'int32')
            batch_idxs = K.arange(0, K.shape(y_true)[0])
            batch_idxs = K.expand_dims(batch_idxs, 1)
            idxs = K.concatenate([batch_idxs, y_true], 1)
            y_true_pred = K.tf.gather_nd(y_pred, idxs)
            y_true_pred = K.expand_dims(y_true_pred, 1)
            y_true_pred_margin = y_true_pred - self.m
            _Z = K.concatenate([y_pred, y_true_pred_margin], 1)
            _Z = _Z * self.s
            logZ = K.logsumexp(_Z, 1, keepdims=True)

            logZ = logZ + K.log(1 - K.exp(self.s * y_true_pred - logZ))
            # now: logZ = K.log((sumexp(Z) - K.exp(self.s * y_true_pred)))
            return -y_true_pred_margin * self.s + logZ

        phi = y_pred - self.m
#         y_true = tf.one_hot(label_batch, self.output_dim)
        adjust_theta = self.s * tf.where(tf.equal(y_true, 1), phi, y_pred)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=adjust_theta)
        return cross_entropy

    def call(self, embeddings, training=None):
        embeddings = keras.backend.l2_normalize(embeddings, -1)
        cos_theta = tf.matmul(embeddings, self.kernel)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)  # for numerical steady
        if self.classification:
            return cos_theta
        return keras.backend.in_train_phase(cos_theta, embeddings, training=training)


if __name__ == '__main__':
    test_FIFOQueue()
