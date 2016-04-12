import tensorflow as tf
import re

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd=None):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    weight_decay = None
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
    return var, weight_decay


def add_loss_summaries(total_loss):
    losses = tf.get_collection('losses')
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)


def add_conv_layer(layer_name, input_images, kernel_attrs, norm_attrs=None, pool_attrs=None):
    tensor_names = list()
    tensors_dict = dict()
    with tf.variable_scope(layer_name) as variable:
        kernel, _ = variable_with_weight_decay(name='kernel', shape=kernel_attrs['shape'],
                                               stddev=kernel_attrs['stddev'])
        tensor_names.append('kernel')
        conv = tf.nn.conv2d(input_images, kernel, kernel_attrs['strides'], padding=kernel_attrs['padding'], name='conv')
        tensor_names.append('conv')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        bias = tf.nn.bias_add(conv, biase)
        relu = tf.nn.relu(bias, name='relu')
        tensor_names.append('relu')
        activation_summary(relu)
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
        kernel, _ = variable_with_weight_decay(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        tensor_names.append('kernel')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        fc = tf.add(tf.matmul(input_images, kernel), biase, name='fc')
        tensor_names.append('fc')
        relu = tf.nn.relu(fc, name='relu')
        tensor_names.append('relu')
        activation_summary(relu)
        for tensor_name in tensor_names:
            tensors_dict.update({layer_name + '_' + tensor_name: eval(tensor_name)})
    return tensors_dict


def add_softmax_layer(layer_name, input_images, kernel_attrs):
    tensor_names = list()
    tensors_dict = dict()
    with tf.variable_scope(layer_name) as variable:
        kernel, _ = variable_with_weight_decay(name='kernel', shape=kernel_attrs['shape'], stddev=kernel_attrs['stddev'])
        tensor_names.append('kernel')
        biase = variable_on_cpu('biase', [kernel_attrs['shape'][-1]], tf.constant_initializer(kernel_attrs['biase']))
        softmax = tf.add(tf.matmul(input_images, kernel), biase, name='softmax')
        tensor_names.append('softmax')
        activation_summary(softmax)
        for tensor_name in tensor_names:
            tensors_dict.update({layer_name + '_' + tensor_name: eval(tensor_name)})
    return tensors_dict


def sig_func(logits):
    saturation = 6
    logits_sig = saturation * (2 / (1 + tf.exp(-2 * logits / saturation)) - 1)
    return logits_sig


def loss_func(logit, logic, lamb):
    logit_pos = tf.boolean_mask(logit, logic)
    logit_neg = tf.boolean_mask(logit, tf.logical_not(logic))
    log_z = tf.log(tf.reduce_mean(tf.exp(logit_neg * lamb)))
    pos_value = tf.reduce_mean(logit_pos * lamb)
    return tf.sub(log_z, pos_value)


def lamb_func(logit, logic, lamb):
    logit_pos = tf.boolean_mask(logit, logic)
    logit_neg = tf.boolean_mask(logit, tf.logical_not(logic))
    logit_neg_exp = tf.exp(logit_neg * lamb)
    z = tf.reduce_mean(logit_neg_exp)
    left = tf.truediv(tf.reduce_mean(logit_neg * logit_neg_exp), z)
    right = tf.reduce_mean(logit_pos)
    return left, right









