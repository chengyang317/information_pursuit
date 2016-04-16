import tensorflow as tf
import re


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_stddev(name, shape, stddev):
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    return var


def add_input_layer(input_attrs):
    layer_name = input_attrs['layer_name']
    with tf.variable_scope(layer_name):
        images = tf.placeholder(dtype=tf.float32, shape=input_attrs['shape'], name='images')
        labels = tf.placeholder(dtype=tf.int32, shape=(input_attrs['shape'][0],), name='labels')
        tensors_dict = {'%s_images' % layer_name: images, '%s_labels' % layer_name: labels}
        return tensors_dict


def add_conv_layer(inputs, kernel_attrs):
    layer_name = kernel_attrs['layer_name']
    with tf.variable_scope(layer_name):
        kernel = variable_with_stddev(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        conv = tf.nn.conv2d(inputs, kernel, kernel_attrs['strides'], padding=kernel_attrs['padding'], name='conv')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        relu = tf.nn.relu(tf.nn.bias_add(conv, biase), name='relu')
        tensors_dict = {'%s_kernel' % layer_name: kernel, '%s_conv' % layer_name: conv, '%s_biase' % layer_name: biase,
                        '%s_relu' % layer_name: relu}
        return tensors_dict


def add_norm_layer(inputs, norm_attrs):
    layer_name = norm_attrs['layer_name']
    with tf.variable_scope(layer_name):
        norm = tf.nn.local_response_normalization(inputs, depth_radius=norm_attrs['depth_radius'],
                                                  bias=norm_attrs['bias'], alpha=norm_attrs['alpha'],
                                                  beta=norm_attrs['beta'], name='norm')
        tensors_dict = {'%s_norm' % layer_name: norm}
        return tensors_dict


def add_pool_layer(inputs, pool_attrs):
    layer_name = pool_attrs['layer_name']
    with tf.variable_scope(layer_name):
        pool = tf.nn.max_pool(input, ksize=pool_attrs['ksize'], strides=pool_attrs['strides'],
                              padding=pool_attrs['padding'], name='pool')
        tensors_dict = {'%s_pool' % layer_name: pool}
        return tensors_dict


def add_full_layer(inputs, kernel_attrs):
    layer_name = kernel_attrs['layer_name']
    with tf.variable_scope(layer_name):
        shape = kernel_attrs['shape']
        if not shape[0]:
            shape[0] = reduce(lambda x, y: x*y, inputs.get_shape().as_list()[1:])
        kernel = variable_with_stddev(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        full = tf.add(tf.matmul(inputs, kernel), biase, name='full')
        relu = tf.nn.relu(full, name='relu')
        tensors_dict = {'%s_kernel' % layer_name: kernel, '%s_full' % layer_name: full, '%s_biase' % layer_name: biase,
                        '%s_relu' % layer_name: relu}
        return tensors_dict


def add_softmax_layer(inputs, kernel_attrs):
    layer_name = kernel_attrs['layer_name']
    with tf.variable_scope(layer_name):
        kernel = variable_with_stddev(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        softmax = tf.add(tf.matmul(inputs, kernel), biase, name='softmax')
        tensors_dict = {'%s_kernel' % layer_name: kernel, '%s_softmax' % layer_name: softmax,
                        '%s_biase' % layer_name: biase}
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











