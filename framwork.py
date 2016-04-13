import tensorflow as tf
import re


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_stddev(name, shape, stddev):
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    return var


def add_conv_layer(layer_name, input_images, kernel_attrs, norm_attrs=None, pool_attrs=None):
    tensor_names = list()
    tensors_dict = dict()
    with tf.variable_scope(layer_name) as variable:
        kernel = variable_with_stddev(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        tensor_names.append('kernel')
        conv = tf.nn.conv2d(input_images, kernel, kernel_attrs['strides'], padding=kernel_attrs['padding'], name='conv')
        tensor_names.append('conv')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        bias = tf.nn.bias_add(conv, biase)
        relu = tf.nn.relu(bias, name='relu')
        tensor_names.append('relu')
        if norm_attrs:
            norm = tf.nn.local_response_normalization(relu, depth_radius=norm_attrs['depth_radius'],
                                                      bias=norm_attrs['bias'], alpha=norm_attrs['alpha'],
                                                      beta=norm_attrs['beta'], name='norm')
            tensor_names.append('norm')
        if pool_attrs:
            input = norm if norm_attrs else relu
            pool = tf.nn.max_pool(input, ksize=pool_attrs['ksize'], strides=pool_attrs['strides'],
                                  padding=pool_attrs['padding'], name='pool')
            tensor_names.append('pool')

        for tensor_name in tensor_names:
            tensors_dict.update({layer_name + '_' + tensor_name: eval(tensor_name)})
    return tensors_dict


def add_full_layer(layer_name, input_images, kernel_attrs):
    tensor_names = list()
    tensors_dict = dict()
    with tf.variable_scope(layer_name) as variable:
        kernel = variable_with_stddev(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        tensor_names.append('kernel')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        fc = tf.add(tf.matmul(input_images, kernel), biase, name='fc')
        tensor_names.append('fc')
        relu = tf.nn.relu(fc, name='relu')
        tensor_names.append('relu')
        for tensor_name in tensor_names:
            tensors_dict.update({layer_name + '_' + tensor_name: eval(tensor_name)})
    return tensors_dict


def add_softmax_layer(layer_name, input_images, kernel_attrs):
    tensor_names = list()
    tensors_dict = dict()
    with tf.variable_scope(layer_name) as variable:
        kernel = variable_with_stddev(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        tensor_names.append('kernel')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        softmax = tf.add(tf.matmul(input_images, kernel), biase, name='softmax')
        tensor_names.append('softmax')
        for tensor_name in tensor_names:
            tensors_dict.update({layer_name + '_' + tensor_name: eval(tensor_name)})
    return tensors_dict


def sig_func(logits):
    saturation = 6
    logits_sig = saturation * (2 / (1 + tf.exp(-2 * logits / saturation)) - 1)
    return logits_sig


def loss_func(logit, logic, lamb):
    def f1():
        return tf.constant(0, dtype=tf.float32)

    def f2():
        logit_pos = tf.boolean_mask(logit, logic)
        logit_neg = tf.boolean_mask(logit, tf.logical_not(logic))
        log_z = tf.log(tf.reduce_mean(tf.exp(logit_neg * lamb)))
        pos_value = tf.reduce_mean(logit_pos * lamb)
        return tf.sub(log_z, pos_value)
    pred = tf.reduce_all(tf.logical_not(logic))
    loss = tf.cond(pred, f1, f2)
    return loss


def lamb_func(logit, logic, lamb):
    logit_pos = tf.boolean_mask(logit, logic)
    logit_neg = tf.boolean_mask(logit, tf.logical_not(logic))
    logit_neg_exp = tf.exp(logit_neg * lamb)
    z = tf.reduce_mean(logit_neg_exp)
    left = tf.truediv(tf.reduce_mean(logit_neg * logit_neg_exp), z)
    right = tf.reduce_mean(logit_pos)
    return left, right









