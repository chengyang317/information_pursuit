# coding=utf-8
__author__ = "Philip_Cheng"

import tensorflow as tf

def init_alex_net_param():
    net_param = list()
    layer1_param = layer.LayerConfig(
        {'type': 'Data', 'crop_size': 227, 'batch_size': 20, 'mirror': True, 'phase': 'Train',
         'top': ['data', 'label'], 'name': 'train_input'})
    layer2_param = layer.LayerConfig(
        {'type': 'Data', 'crop_size': 227, 'batch_size': 20, 'mirror': False, 'phase': 'TEST',
         'top': ['data', 'label'], 'name': 'test_input'})
    layer3_param = layer.LayerConfig(
        {'type': 'Convolution', 'kernel_size': [11, 11, 3, 96], 'stride_size': [1, 4, 4, 1],
         'weight_filler': {'type': 'gaussian', 'std': 0.01},
         'bias_filler': {'type': 'constant', 'value': 0},
         'bottom': ['data'], 'top': ['conv1'], 'name': 'conv1'})
    layer4_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv1'], 'top': ['relu1'], 'name': 'relu1'})
    layer5_param = layer.LayerConfig({'type': 'Lrn', 'local_size': 5, 'alpha': 0.0001, 'beta': 0.75,
                                      'bottom': ['relu1'], 'top': ['lrn1'], 'name': 'lrn1'})
    layer6_param = layer.LayerConfig({'type': 'Pool', 'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1],
                                      'bottom': ['lrn1'], 'top': ['pool1'], 'name': 'pool1'})
    layer7_param = layer.LayerConfig(
        {'type': 'Convolution', 'kernel_size': [5, 5, 96, 256], 'stride_size': [1, 2, 2, 1],
         'weight_filler': {'type': 'gaussian', 'std': 0.01},
         'bias_filler': {'type': 'constant', 'value': 0.1},
         'bottom': ['pool1'], 'top': ['conv2'], 'name': 'conv2'})
    layer8_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv2'], 'top': ['relu2'], 'name': 'relu2'})
    layer9_param = layer.LayerConfig({'type': 'Lrn', 'local_size': 5, 'alpha': 0.0001, 'beta': 0.75,
                                      'bottom': ['relu2'], 'top': ['lrn21'], 'name': 'lrn2'})
    layer10_param = layer.LayerConfig({'type': 'Pool', 'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1],
                                       'bottom': ['lrn2'], 'top': ['pool2'], 'name': 'pool2'})
    layer11_param = layer.LayerConfig(
        {'type': 'Convolution', 'kernel_size': [3, 3, 256, 384], 'stride_size': [1, 1, 1, 1],
         'weight_filler': {'type': 'gaussian', 'std': 0.01},
         'bias_filler': {'type': 'constant', 'value': 0},
         'bottom': ['pool2'], 'top': ['conv3'], 'name': 'conv3'})
    layer12_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv3'], 'top': ['relu3'], 'name': 'relu3'})
    layer13_param = layer.LayerConfig(
        {'type': 'Convolution', 'kernel_size': [3, 3, 384, 384], 'stride_size': [1, 1, 1, 1],
         'weight_filler': {'type': 'gaussian', 'std': 0.01},
         'bias_filler': {'type': 'constant', 'value': 0.1},
         'bottom': ['relu3'], 'top': ['conv4'], 'name': 'conv4'})
    layer14_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv4'], 'top': ['relu4'], 'name': 'relu4'})
    layer15_param = layer.LayerConfig(
        {'type': 'Convolution', 'kernel_size': [3, 3, 384, 256], 'stride_size': [1, 1, 1, 1],
         'weight_filler': {'type': 'gaussian', 'std': 0.01},
         'bias_filler': {'type': 'constant', 'value': 0.1},
         'bottom': ['relu4'], 'top': ['conv5'], 'name': 'conv5'})
    layer16_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv5'], 'top': ['relu5'], 'name': 'relu5'})
    layer17_param = layer.LayerConfig({'type': 'Pool', 'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1],
                                       'bottom': ['relu5'], 'top': ['pool5'], 'name': 'pool'})
    layer18_param = layer.LayerConfig({'type': 'InnerProduct', 'kernel_size': [None, 4096],
                                       'weight_filler': {'type': 'gaussian', 'std': 0.005},
                                       'bias_filler': {'type': 'constant', 'value': 0.1},
                                       'bottom': ['pool5'], 'top': ['fc6'], 'name': 'fc6'})
    layer19_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['fc6'], 'top': ['relu6'], 'name': 'relu6'})
    layer20_param = layer.LayerConfig({'type': 'Dropout', 'dropout_ratio': 0.5, 'phase': 'TRAIN',
                                       'bottom': ['relu6'], 'top': ['drop6'], 'name': 'drop6'})
    layer21_param = layer.LayerConfig({'type': 'InnerProduct', 'kernel_size': [4096, 4096],
                                       'weight_filler': {'type': 'gaussian', 'std': 0.005},
                                       'bias_filler': {'type': 'constant', 'value': 0.1},
                                       'bottom': ['drop6'], 'top': ['fc7'], 'name': 'fc7'})
    layer22_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['fc7'], 'top': ['relu7'], 'name': 'relu7'})
    layer23_param = layer.LayerConfig({'type': 'Dropout', 'dropout_ratio': 0.5, 'phase': 'TRAIN',
                                       'bottom': ['relu7'], 'top': ['drop7'], 'name': 'drop7'})
    layer24_param = layer.LayerConfig({'type': 'InnerProduct', 'kernel_size': [4096, 2],
                                       'weight_filler': {'type': 'gaussian', 'std': 0.01},
                                       'bias_filler': {'type': 'constant', 'value': 0},
                                       'bottom': ['drop7'], 'top': ['fc8'], 'name': 'fc8'})
    layer25_param = layer.LayerConfig({'type': 'InformationPursueLoss'})
    layer25_param = layer.LayerConfig({'type': 'Accuracy', 'bottom': ['fc8', 'label'],
                                       'top': ['accuracy'], 'name': 'accuracy', 'phase': 'TEST'})
    layer26_param = layer.LayerConfig({'type': 'SoftmaxWithLoss', 'bottom': ['fc8', 'label'],
                                       'top': ['loss'], 'name': 'loss', 'phase': 'TRAIN'})
    for index in range(1, 27):
        net_param.append(eval('layer%d_param' % index))

class Alex_net(net.Net):
    def __init__(self):
        self.losses = list()
        self.cpu_device = '/cpu:0'
        self.net_parameter = list()
        self.outputs = dict()
        pass

    def init_net_param(self):
        net_param = self.net_parameter
        layer1_param = layer.LayerConfig({'type': 'Data', 'crop_size': 227, 'batch_size': 20, 'mirror': True, 'phase': 'Train',
                       'top': ['data', 'label'], 'name': 'train_input'})
        layer2_param = layer.LayerConfig({'type': 'Data', 'crop_size': 227, 'batch_size': 20, 'mirror': False, 'phase': 'TEST',
                       'top': ['data', 'label'], 'name': 'test_input'})
        layer3_param = layer.LayerConfig({'type': 'Convolution', 'kernel_size': [11, 11, 3, 96], 'stride_size': [1, 4, 4, 1],
                       'weight_filler': {'type': 'gaussian', 'std': 0.01},
                       'bias_filler': {'type': 'constant', 'value': 0},
                       'bottom': ['data'], 'top': ['conv1'], 'name': 'conv1'})
        layer4_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv1'], 'top': ['relu1'], 'name': 'relu1'})
        layer5_param = layer.LayerConfig({'type': 'Lrn', 'local_size': 5, 'alpha': 0.0001, 'beta': 0.75,
                      'bottom': ['relu1'], 'top': ['lrn1'], 'name': 'lrn1'})
        layer6_param = layer.LayerConfig({'type': 'Pool', 'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1],
                       'bottom': ['lrn1'], 'top': ['pool1'], 'name': 'pool1'})
        layer7_param = layer.LayerConfig({'type': 'Convolution', 'kernel_size': [5, 5, 96, 256], 'stride_size': [1, 2, 2, 1],
                       'weight_filler': {'type': 'gaussian', 'std': 0.01},
                       'bias_filler': {'type': 'constant', 'value': 0.1},
                       'bottom': ['pool1'], 'top': ['conv2'], 'name': 'conv2'})
        layer8_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv2'], 'top': ['relu2'], 'name': 'relu2'})
        layer9_param = layer.LayerConfig({'type': 'Lrn', 'local_size': 5, 'alpha': 0.0001, 'beta': 0.75,
                      'bottom': ['relu2'], 'top': ['lrn21'], 'name': 'lrn2'})
        layer10_param = layer.LayerConfig({'type': 'Pool', 'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1],
                       'bottom': ['lrn2'], 'top': ['pool2'], 'name': 'pool2'})
        layer11_param = layer.LayerConfig({'type': 'Convolution', 'kernel_size': [3, 3, 256, 384], 'stride_size': [1, 1, 1, 1],
                       'weight_filler': {'type': 'gaussian', 'std': 0.01},
                       'bias_filler': {'type': 'constant', 'value': 0},
                       'bottom': ['pool2'], 'top': ['conv3'], 'name': 'conv3'})
        layer12_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv3'], 'top': ['relu3'], 'name': 'relu3'})
        layer13_param = layer.LayerConfig({'type': 'Convolution', 'kernel_size': [3, 3, 384, 384], 'stride_size': [1, 1, 1, 1],
                       'weight_filler': {'type': 'gaussian', 'std': 0.01},
                       'bias_filler': {'type': 'constant', 'value': 0.1},
                       'bottom': ['relu3'], 'top': ['conv4'], 'name': 'conv4'})
        layer14_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv4'], 'top': ['relu4'], 'name': 'relu4'})
        layer15_param = layer.LayerConfig({'type': 'Convolution', 'kernel_size': [3, 3, 384, 256], 'stride_size': [1, 1, 1, 1],
                       'weight_filler': {'type': 'gaussian', 'std': 0.01},
                       'bias_filler': {'type': 'constant', 'value': 0.1},
                       'bottom': ['relu4'], 'top': ['conv5'], 'name': 'conv5'})
        layer16_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['conv5'], 'top': ['relu5'], 'name': 'relu5'})
        layer17_param = layer.LayerConfig({'type': 'Pool', 'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1],
                       'bottom': ['relu5'], 'top': ['pool5'], 'name': 'pool'})
        layer18_param = layer.LayerConfig({'type': 'InnerProduct', 'kernel_size': [None, 4096],
                     'weight_filler': {'type': 'gaussian', 'std': 0.005},
                     'bias_filler': {'type': 'constant', 'value': 0.1},
                     'bottom': ['pool5'], 'top': ['fc6'], 'name': 'fc6'})
        layer19_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['fc6'], 'top': ['relu6'], 'name': 'relu6'})
        layer20_param = layer.LayerConfig({'type':'Dropout', 'dropout_ratio': 0.5, 'phase': 'TRAIN',
                       'bottom': ['relu6'], 'top': ['drop6'], 'name': 'drop6'})
        layer21_param = layer.LayerConfig({'type': 'InnerProduct', 'kernel_size': [4096, 4096],
                     'weight_filler': {'type': 'gaussian', 'std': 0.005},
                     'bias_filler': {'type': 'constant', 'value': 0.1},
                     'bottom': ['drop6'], 'top': ['fc7'], 'name': 'fc7'})
        layer22_param = layer.LayerConfig({'type': 'Relu', 'bottom': ['fc7'], 'top': ['relu7'], 'name': 'relu7'})
        layer23_param = layer.LayerConfig({'type':'Dropout', 'dropout_ratio': 0.5, 'phase': 'TRAIN',
                       'bottom': ['relu7'], 'top': ['drop7'], 'name': 'drop7'})
        layer24_param = layer.LayerConfig({'type': 'InnerProduct', 'kernel_size': [4096, 2],
                     'weight_filler': {'type': 'gaussian', 'std': 0.01},
                     'bias_filler': {'type': 'constant', 'value': 0},
                     'bottom': ['drop7'], 'top': ['fc8'], 'name': 'fc8'})
        layer25_param = layer.LayerConfig({'type': 'Accuracy', 'bottom': ['fc8', 'label'],
                                          'top': ['accuracy'], 'name': 'accuracy', 'phase': 'TEST'})
        layer26_param = layer.LayerConfig({'type': 'SoftmaxWithLoss', 'bottom': ['fc8', 'label'],
                                          'top': ['loss'], 'name': 'loss', 'phase': 'TRAIN'})
        for index in range(1, 27):
            net_param.append(eval('layer%d_param' % index))

    def build_train_network(self):
        net_param = self.net_parameter
        for layer_id, layer_param in enumerate(net_param):
            phase = layer_param.get('phase', None)
            if phase == 'TEST':
                continue
            self.build_layer(layer_param)

    def build_layer(self, layer_param):
        layer_func = layer_function.get_layer_function(layer_param)
        inputs = []
        layer_name = layer_param.name
        with tf.name_scope(layer_name):
            if hasattr(layer_param, 'bottom'):
                bottom_list = layer_param.bottom
                for bottom in bottom_list:
                    if not self.outputs.has_key('bottom'):
                        raise KeyError('Cant find %s in outputs' % bottom)
                    inputs.append(self.outputs[bottom])

            outputs = layer_func(inputs, layer_param)
            self.outputs.update(outputs)






    def add_to_losses(self, loss):
        self.losses.append(loss)

    def _activation_summary(self, x):
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

    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor
        """
        with tf.device(self.cpu_device):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd):
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
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            self.add_to_losses(weight_decay)
        return var

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('%d_losses' % index)
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(l.op.name +' (raw)', l)
            tf.scalar_summary(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def alex_inference(self, images, dropout):
        """
        Args:
            images: Images returned from distorted_inputs() or inputs().
        Returns:
            Logits.
        """
        # We instantiate all variables using tf.get_variable() instead of
        # tf.Variable() in order to share variables across multiple GPU training runs.
        # If we only ran this model on a single GPU, we could simplify this function
        # by replacing all instances of tf.get_variable() with tf.Variable().
        #
        # conv1
        with tf.variable_scope("conv1") as scope:
            kernel = self._variable_with_weight_decay(name='weights', shape=[11, 11, 3, 96], stddev=1e-1, wd=0.004)
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv1)

        # norm1
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')

        # pool1
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        # conv2
        with tf.variable_scope("conv2") as scope:
            kernel = self._variable_with_weight_decay("weights", shape=[5, 5, 96, 256], stddev=1e-1, wd=0.004, index=index)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv2)

        # norm2
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm2')

        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        # conv3
        with tf.variable_scope("conv3") as scope:
            kernel = self._variable_with_weight_decay("weights", shape=[3, 3, 256, 384], stddev=1e-1, wd=0.004, index=index)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv3)

        # conv4
        with tf.variable_scope("conv4") as scope:
            kernel = self._variable_with_weight_decay("weights", shape=[3, 3, 384, 384], stddev=1e-1, wd=0.004, index=index)
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv4)

        # conv5
        with tf.variable_scope("conv5") as scope:
            kernel = self._variable_with_weight_decay("weights", shape=[3, 3, 384, 256], stddev=1e-1, wd=0.004, index=index)
            conv = tf.nn.conv2d(conv4, kernel, [1, 3, 3, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope.name)
            self._activation_summary(conv5)

        # pool5
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [256, self.num_classes], stddev=1e-1, wd=0.004, index=index)
            biases = self._variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(pool5, weights), biases, name=scope.name)
            self._activation_summary(softmax_linear)

        return softmax_linear