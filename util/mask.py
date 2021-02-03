"""Embedding layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.topology import Layer
from keras.legacy import interfaces
from keras.utils.generic_utils import to_list
from tensorflow.python.framework import dtypes

import tensorflow as tf
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.backend.tensorflow_backend import dropout


class MaskEmbedding(Layer):
    """Turns positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    This layer can only be used as the first layer in a model.

    # Example

    ```python
      model = Sequential()
      model.add(Embedding(1000, 64, input_length=10))
      # the model will take as input an integer matrix of size (batch, input_length).
      # the largest integer (i.e. word index) in the input should be
      # no larger than 999 (vocabulary size).
      # now model.output_shape == (None, 10, 64), where None is the batch dimension.

      input_array = np.random.randint(1000, size=(32, 10))

      model.compile('rmsprop', 'mse')
      output_array = model.predict(input_array)
      assert output_array.shape == (32, 10, 64)
    ```

    # Arguments
        input_dim: int > 0. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: int >= 0. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings` matrix
            (see [initializers](../initializers.md)).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix
            (see [regularizer](../regularizers.md)).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix
            (see [constraints](../constraints.md)).
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
        - [A Theoretically Grounded Application of Dropout in
           Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_embedding_support
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(MaskEmbedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.supports_masking = True
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.built = True

#     def compute_mask(self, inputs, mask = None):
#         output_mask = K.not_equal(inputs, -1)
#         return output_mask

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            # input_length can be tuple if input is 3D or higher
            in_lens = to_list(self.input_length, allow_tuple=True)
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError(
                    '"input_length" is %s, but received input has shape %s' %
                    (str(self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError(
                            '"input_length" is %s, but received input has shape %s' %
                            (str(self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer':
                      initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer':
                      regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint':
                      constraints.serialize(self.embeddings_constraint),
                  'input_length': self.input_length}
        base_config = super(MaskEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskMinusOne(Layer):

    def __init__(self, **kwargs):
        super(MaskMinusOne, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskMinusOne, self).build(input_shape)

    def call(self, x):

        ones = tf.ones(x.shape, dtypes.int32)

        condition = tf.add(x, ones)

        indices = tf.where(condition)

        indices = tf.reshape(indices, [-1])

        output = tf.gather(x, indices)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape


class GRUEmbedding(Layer):

    def __init__(self, dimensionEmbedding, word_size, **kwargs):
        super(GRUEmbedding, self).__init__(**kwargs)
#         self.embedding = Embedding(dimensionEmbedding, word_size, mask_zero=True)
#         Embedding.__init__(self, dimensionEmbedding, word_size, mask_zero=True)
#         GRU.__init__(self, word_size, **kwargs)

#         self.gru = GRU(word_size)
        self.word_size = word_size
        self.dimensionEmbedding = dimensionEmbedding

    def build(self, input_shape):
        self.buildEmbedding()
#         GRU.build(self, (self.word_size,))

        self.built = True

    def buildEmbedding(self):
        self.embeddings = self.add_weight(
            shape=(self.dimensionEmbedding, self.word_size),
            initializer='uniform',
            name='embeddings',
            dtype='float32')

        self.built = True

    def call(self, x):
        embedding = K.gather(self.embeddings, x)
        output = tf.reduce_mean(embedding, 1)

        return output
#         embedding = self.embedding(x)
        gru = GRU.call(self, embedding)
#         x = tf.convert_to_tensor(x)
#         K.learning_phase()
        gru._uses_learning_phase = False;

        return gru

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.word_size)
