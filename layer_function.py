# coding=utf-8
__author__ = "Philip_Cheng"

import registry
import six
import tensorflow as tf
import framwork

_layer_functions_registry = registry.Registry("layer_functions")


class RegisterLayerFunction(object):
    def __init__(self, layer_type):
        """Creates a new decorator with `op_type` as the Operation type.
        Args:
          op_type: The string type of an operation. This corresponds to the
            `OpDef.name` field for the proto that defines the operation.
        """
        if not isinstance(layer_type, six.string_types):
            raise TypeError("op_type must be a string")
        self._layer_type = layer_type

    def __call__(self, f):
        """Registers the function `f` as gradient function for `op_type`."""
        _layer_functions_registry.register(f, self._layer_type)
        return f


def NoGradient(op_type):
    """Specifies that ops of type `op_type` do not have a defined gradient.
    This function is only used when defining a new op type. It may be
    used for ops such as `tf.size()` that are not differentiable.  For
    example:
    ```python
    tf.NoGradient("Size")
    ```
    Args:
      op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
    Raises:
      TypeError: If `op_type` is not a string.
    """
    if not isinstance(op_type, six.string_types):
        raise TypeError("op_type must be a string")
    _layer_functions_registry.register(None, op_type)


def get_layer_function(layer_param):
    """Returns the function that computes gradients for "op"."""
    try:
        layer_type = layer_param.get_attr("type")
    except ValueError:
        print('Cant find type in layer config')
    return _layer_functions_registry.lookup(layer_type)


@RegisterLayerFunction('Data')
def data_layer_function(inputs, layer_param):
    batch_size = layer_param.batch_size
    crop_size = layer_param.crop_size
    if not hasattr(layer_param, 'depth'):
        depth = 3
    else:
        depth = layer_param.depth
    top_names = layer_param.top
    outputs = dict()
    for name in top_names:
        if name == 'data':
            outputs[name] = tf.placeholder(dtype=tf.float32, shape=(batch_size, crop_size, crop_size, depth), name=name)
        else:
            outputs[name] = tf.placeholder(dtype=tf.int8, shape=(batch_size,), name=name)
    return outputs

@RegisterLayerFunction('InformationPursueData')
def information_pursue_data_layer_function(inputs, layer_param):
    batch_size = layer_param.batch_size
    image_size = layer_param.image_size
    if not hasattr(layer_param, 'depth'):
        depth = 3
    else:
        depth = layer_param.depth
    top_name = layer_param.top[0]
    outputs = dict()
    outputs[top_name] = tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, image_size, image_size, depth), name=top_name)
    return outputs

@RegisterLayerFunction('Convolution')
def convolution_layer_function(inputs, layer_param):
    if len(inputs) != 1:
        raise ValueError('inputs is not legal')
    input = inputs[0]
    kernel_size = layer_param.kernel_size
    stride_size = layer_param.stride_size
    weight_filler = layer_param.weight_filler
    bias_filler = layer_param.bias_filler
    weight_decay = weight_filler.get('weight_decay', None)
    bias_size = [kernel_size[-1]]
    top_name = layer_param.top[0]
    outputs = dict()
    if weight_filler['type'] == 'gaussian':
        kernel, weight_decay = framwork._variable_with_weight_decay(name='kernel_weight',
                    shape=kernel_size, stddev=weight_filler['std'], wd=weight_decay)
    else:
        raise ValueError('Unknown convolution type')
    if bias_filler['type'] == 'constant':
        bias = framwork._variable_on_cpu('biases', bias_size, tf.constant_initializer(bias_filler['value']))

    outputs[top_name] = tf.nn.bias_add(tf.nn.conv2d(input=input, filter=kernel, strides=stride_size,
                                                    padding='SAME'), bias, name=top_name)
    if weight_decay:
        outputs['%s_weight_decay' % layer_param.name] = weight_decay
    return outputs

@RegisterLayerFunction('Relu')
def relu_layer_function(inputs, layer_param):
    if len(inputs) != 1:
        raise ValueError('inputs is not legal')
    input = inputs[0]
    outputs = dict()
    top_name = layer_param.top[0]
    outputs[top_name] = tf.nn.relu(features=input, name=top_name)
    return outputs

@RegisterLayerFunction('Lrn')
def lrn_layer_function(inputs, layer_param):
    if len(inputs) != 1:
        raise ValueError('inputs is not legal')
    input = inputs[0]
    outputs = dict()
    top_name = layer_param.top[0]
    local_size = layer_param.local_size
    alpha = layer_param.alpha
    beta = layer_param.beta
    outputs[top_name] = tf.nn.local_response_normalization(input=input, depth_radius=local_size,
                                            bias=1.0, alpha=alpha, beta=beta, name=top_name)
    return outputs

@RegisterLayerFunction('Pool')
def pool_layer_function(inputs, layer_param):
    if len(inputs) != 1:
        raise ValueError('inputs is not legal')
    input = inputs[0]
    outputs = dict()
    top_name = layer_param.top[0]
    kernel_size = layer_param.kernel_size
    stride_size = layer_param.stride_size
    outputs[top_name] = tf.nn.max_pool(input, ksize=kernel_size, strides=stride_size,
                                       padding='SAME', name=top_name)
    return outputs


@RegisterLayerFunction('InnerProduct')
def inner_product_layer_function(inputs, layer_param):
    if len(inputs) != 1:
        raise ValueError('inputs is not legal')
    input = inputs[0]
    outputs = dict()
    top_name = layer_param.top[0]
    kernel_size = layer_param.kernel_size
    weight_filler = layer_param.weight_filler
    bias_filler = layer_param.bias_filler
    weight_decay = weight_filler.get('weight_decay', None)
    if not kernel_size[0]:
        kernel_size[0] = input.get_shape().ndims
    bias_size = [kernel_size[-1]]
    if weight_filler['type'] == 'gaussian':
        kernel, weight_decay = framwork._variable_with_weight_decay(name='inner_weight',
                    shape=kernel_size, stddev=weight_filler['std'], wd=weight_decay)
    else:
        raise ValueError('Unknown convolution type')
    if bias_filler['type'] == 'constant':
        bias = framwork._variable_on_cpu('biases', bias_size, tf.constant_initializer(bias_filler['value']))

    input = tf.reshape(input, kernel_size[0])
    outputs[top_name] = tf.add(tf.matmul(input, kernel), bias, name=top_name)
    return outputs


@RegisterLayerFunction('Dropout')
def dropout_layer_function(inputs, layer_param):
    if len(inputs) != 1:
        raise ValueError('inputs is not legal')
    input = inputs[0]
    outputs = dict()
    top_name = layer_param.top[0]
    dropout_ratio = layer_param.dropout_ration
    outputs[top_name] = tf.nn.dropout(input, dropout_ratio, name=top_name)
    return outputs


