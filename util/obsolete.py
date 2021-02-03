

from distutils.version import StrictVersion
import keras
import keras.backend as K

if StrictVersion(keras.__version__.split('-')[0]) >= StrictVersion('2.2.4'):
    from keras.backend.common import normalize_data_format
    from keras.layers.wrappers import TimeDistributed
    from keras.activations import softmax
else:
    from keras.utils.conv_utils import normalize_data_format  # @UnresolvedImport
    from keras.layers.wrappers import Wrapper
    from keras.utils.generic_utils import has_arg
    from keras.layers import InputSpec

    def softmax(x, axis=-1):
        """Softmax activation function.
    
        # Arguments
            x: Input tensor.
            axis: Integer, axis along which the softmax normalization is applied.
    
        # Returns
            Tensor, output of softmax transformation.
    
        # Raises
            ValueError: In case `dim(x) == 1`.
        """
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s

    class TimeDistributed(Wrapper):

        def __init__(self, layer, **kwargs):
            super(TimeDistributed, self).__init__(layer, **kwargs)
            self.supports_masking = True

        def _get_shape_tuple(self, init_tuple, tensor, start_idx, int_shape=None):
            # replace all None in int_shape by K.shape
            if int_shape is None:
                int_shape = K.int_shape(tensor)[start_idx:]
            if not any(not s for s in int_shape):
                return init_tuple + int_shape
            tensor_shape = K.shape(tensor)
            int_shape = list(int_shape)
            for i, s in enumerate(int_shape):
                if not s:
                    int_shape[i] = tensor_shape[start_idx + i]
            return init_tuple + tuple(int_shape)

        def build(self, input_shape):
            assert len(input_shape) >= 3
            self.input_spec = InputSpec(shape=input_shape)
            child_input_shape = (input_shape[0],) + input_shape[2:]
            if not self.layer.built:
                self.layer.build(child_input_shape)
                self.layer.built = True
            super(TimeDistributed, self).build()

        def compute_output_shape(self, input_shape):
            child_input_shape = (input_shape[0],) + input_shape[2:]
            child_output_shape = self.layer.compute_output_shape(child_input_shape)
            timesteps = input_shape[1]
            return (child_output_shape[0], timesteps) + child_output_shape[1:]

        def call(self, inputs, training=None, mask=None):
            kwargs = {}
            if has_arg(self.layer.call, 'training'):
                kwargs['training'] = training
            uses_learning_phase = False

            input_shape = K.int_shape(inputs)
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(inputs)[1]
            inner_input_shape = self._get_shape_tuple((-1,), inputs, 2)
            # Shape: (num_samples * timesteps, ...). And track the

            inputs = K.reshape(inputs, inner_input_shape)
            # (num_samples * timesteps, ...)
            if has_arg(self.layer.call, 'mask') and mask is not None:
                inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
                kwargs['mask'] = K.reshape(mask, inner_mask_shape)
            y = self.layer.call(inputs, **kwargs)
            if hasattr(y, '_uses_learning_phase'):
                uses_learning_phase = y._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape)
            output_shape = self._get_shape_tuple(
                (-1, input_length), y, 1, output_shape[2:])
            y = K.reshape(y, output_shape)

            # Apply activity regularizer if any:
            if (hasattr(self.layer, 'activity_regularizer') and
               self.layer.activity_regularizer is not None):
                regularization_loss = self.layer.activity_regularizer(y)
                self.add_loss(regularization_loss, inputs)

            if uses_learning_phase:
                y._uses_learning_phase = True
            return y

        def compute_mask(self, *_):
            return None
