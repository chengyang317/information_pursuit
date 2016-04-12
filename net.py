# coding=utf-8
__author__ = "Philip_Cheng"


import tensorflow as tf
import numpy as np
import framwork
import dataset
import Queue
import os


class InforNet(object):
    # network class for Information Pursue Model
    def __init__(self, batch_size, network_percent, image_classes):
        self.train_dir = '/scratch/dataset/information_pursue'
        self.image_shape = (227, 227, 3)
        self.weight_decay = 0.1
        self.learning_rate = 1e-1
        self.net_train_maxstep = 5
        self.image_classes = image_classes
        self.network_percent = network_percent
        self.dataset_percent = 0.05
        self.batch_size = batch_size
        self.train_que = Queue.Queue(maxsize=1000)
        self.test_que = Queue.Queue(maxsize=1000)
        self.train_reader = dataset.Reader(self.train_que, os.path.join(self.train_dir, 'train_caltech_lmdb_%s' % str(self.dataset_percent)))
        self.test_reader = dataset.Reader(self.test_que, os.path.join(self.train_dir, 'test_caltech_lmdb_%s' % str(self.dataset_percent)))
        self.devices = ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']
        self.net_device = self.devices[3]
        self.net_tensors = dict()
        self.net_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.log_device_placement)
        config.gpu_options.allocator_type = 'BFC'
        self.net_sess = tf.Session(graph=self.net_graph, config=config)

    def net_placeholders(self):
        with tf.name_scope('net_placeholders'):
            shape = (self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2])
            images_placeholder = tf.placeholder(dtype=tf.float32, shape=shape, name='images_placeholder')
            labels_placeholder = tf.placeholder(dtype=tf.int32, shape=shape[0], name='labels_placeholder')
            lambda_placeholder = tf.placeholder(dtype=tf.float32, name='lambda_placeholder')
        tensors_dict = {'images_placeholder': images_placeholder, 'labels_placeholder': labels_placeholder,
                        'lambda_placeholder': lambda_placeholder}
        return tensors_dict

    def net_logits(self, images):
        network_percent = self.network_percent
        tensors_dict = dict()
        with tf.variable_scope('net_logits'):
            # first layer
            input_image = images
            layer_name = 'conv1'
            kernel_attrs = {'shape': [11, 11, 3, int(96 * network_percent)], 'stddev': 1e-2, 'strides': [1, 4, 4, 1],
                            'padding': 'SAME', 'biase': 0.0}
            norm_attrs = {'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75}
            pool_attrs = {'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'}
            tensors_dict.update(framwork.add_conv_layer(layer_name, input_image, kernel_attrs=kernel_attrs,
                                                        norm_attrs=norm_attrs, pool_attrs=pool_attrs))
            # second layer
            input_image = tensors_dict[layer_name + '_pool']
            layer_name = 'conv2'
            update_kernel = {'shape': [5, 5, int(96 * network_percent), int(256 * network_percent)],
                             'strides': [1, 2, 2, 1], 'bias': 0.1}
            kernel_attrs.update(update_kernel)
            tensors_dict.update(framwork.add_conv_layer(layer_name, input_image, kernel_attrs=kernel_attrs,
                                                        norm_attrs=norm_attrs, pool_attrs=pool_attrs))
            # third layer
            input_image = tensors_dict[layer_name + '_pool']
            layer_name = 'conv3'
            update_kernel = {'shape': [3, 3, int(256 * network_percent), int(256 * network_percent)],
                             'strides': [1, 1, 1, 1]}
            kernel_attrs.update(update_kernel)
            tensors_dict.update(framwork.add_conv_layer(layer_name, input_image, kernel_attrs=kernel_attrs,
                                                        pool_attrs=pool_attrs))
            # fully connected layer
            input_image = tensors_dict[layer_name + '_pool']
            input_image_shape = input_image.get_shape().as_list()
            input_image_ndims = input_image_shape[1] * input_image_shape[2] * input_image_shape[3]
            input_image = tf.reshape(input_image, (input_image_shape[0], input_image_ndims))
            layer_name = 'full'
            update_kernel = {'shape': [input_image_ndims, int(4096 * network_percent)], 'stddev': 0.005}
            kernel_attrs.update(update_kernel)
            tensors_dict.update(framwork.add_full_layer(layer_name, input_image, kernel_attrs=kernel_attrs))
            # softmax
            input_image = tensors_dict[layer_name + '_relu']
            layer_name = 'softmax'
            update_kernel = {'shape': [int(4096 * network_percent, self.image_classes)], 'stddev': 1e-2}
            kernel_attrs.update(update_kernel)
            tensors_dict.update(framwork.add_softmax_layer(layer_name, input_image, kernel_attrs=kernel_attrs))
            # add a alias name to logits
            tensors_dict.update({'logits': tensors_dict[layer_name + '_softmax']})
        return tensors_dict

    def net_loss(self, logits, labels, lambs):
        with tf.name_scope('net_loss'):
            # put a sigfunction on logits and then transpose
            logits = tf.transpose(framwork.sig_func(logits))
            # according to the labels, erase rows which is not in labels
            labels_unique, _ = tf.unique(labels)
            labels_num = tf.size(labels_unique)
            logits = tf.gather(logits, indices=labels_unique)
            lambs = tf.gather(lambs, indices=labels_unique)
            # set the value of each row to True when it occurs in labels
            templete = tf.tile(tf.expand_dims(labels_unique, dim=1), [1, self.batch_size])
            labels_expand = tf.tile(tf.expand_dims(labels, dim=0), [labels_num, 1])
            indict_logic = tf.equal(labels_expand, templete)
            # split the tensor along rows
            logit_list = tf.split(0, labels_num, templete)
            indict_logic_list = tf.split(0, labels_num, indict_logic)
            lambda_list = tf.split(0, self.image_classes, lambs)
            
            loss_list = map(framwork.loss_func(), logit_list, indict_logic_list, lambda_list)
            losses = tf.add_n(loss_list)
            tensors_dict = {'labels_unique': labels_unique, 'templete': templete, 'logits_sig_trans': logits,
                            'net_losses': losses, 'indict_logic': indict_logic}
        return tensors_dict
    
    def net_total_loss(self, net_loss):
        with tf.name_scope('net_total_loss'):
            net_tensors = self.net_tensors
            tensors_dict = {}
            weight_loss = list()
            for layer_name in ['conv1', 'conv2', 'conv3', 'full', 'softmax']:
                kernel = net_tensors[layer_name + '_kernel']
                kernel_loss = tf.mul(tf.nn.l2_loss(kernel), self.weight_decay, name=layer_name + '_kernel_loss')
                weight_loss.append(kernel_loss)
                tensors_dict.update({layer_name + '_kernel_loss': kernel_loss})
            weight_loss.append(net_loss)
            net_total_loss = tf.add_n(weight_loss, name='total_loss')
            tensors_dict.update({'net_total_loss': net_total_loss})
            return tensors_dict

    def net_train(self, net_total_loss):
        with tf.name_scope('net_train'):
            net_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            net_grads = net_optimizer.compute_gradients(net_total_loss)
            net_train_op = net_optimizer.apply_gradients(net_grads)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)
            # Add histograms for gradients.
            for grad, var in net_grads:
                if grad:
                    tf.histogram_summary(var.op.name + '/gradients', grad)
        tensors_dict = {'net_optimizer': net_optimizer, 'net_grads': net_grads,
                        'net_train_op': net_train_op}
        return tensors_dict

    def net_other(self):
        with tf.name_scope('net_other'):
            net_init_op = tf.initialize_all_variables()
            net_saver = tf.train.Saver(tf.all_variables())
            feautre_summary_op = tf.merge_all_summaries()
            net_summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=self.net_sess.graph_def)
        tensors_dict = {'net_init_op': net_init_op, 'net_saver': net_saver,
                        'net_summary_op': feautre_summary_op, 'net_summary_writer': net_summary_writer}
        return tensors_dict
    
    def build_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            net_tensors = self.net_tensors
            # create placeholders needed
            net_placeholders_tensors = self.net_placeholders()
            net_tensors.update(net_placeholders_tensors)
            # create logits tensor
            images_placeholder = net_tensors['images_placeholder']
            net_logits_tensors = self.net_logits(images_placeholder)
            net_tensors.update(net_logits_tensors)
            # create loss tensors
            lambs = net_tensors['lambda_placeholder']
            logits = net_tensors['logits']
            labels = net_tensors['labels_placeholder']
            net_loss_tensors = self.net_loss(logits, labels, lambs)
            net_tensors.update(net_loss_tensors)
            # create total loss(adding the loss of weight)
            net_loss = net_loss_tensors['net_loss']
            net_total_loss_tensors = self.net_total_loss(net_loss)
            net_tensors.update(net_total_loss_tensors)
            # create train tensor
            net_total_loss = net_total_loss_tensors['net_total_loss']
            net_train_tensors = self.net_train(net_total_loss)
            net_tensors.update(net_train_tensors)
            # create endwork tensor
            net_other_tensors = self.net_other()
            net_tensors.update(net_other_tensors)

    def fetch_datas(self, que):
        image_datas = list()
        for i in self.batch_size:
            image_data = que.get()
            image_datas.append(image_data)
        que.task_done()
        datas = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)
        labels = np.empty(self.batch_size, dtype=np.int32)
        for i in self.batch_size:
            datas[i, :] = image_data['data'].astype(dtype=np.float32)
            labels[i] = int(image_data['labels'])
        return datas, labels

    def train_network(self, lamb):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            net_tensors = self.net_tensors

            images_placeholder = net_tensors['images_placeholder']
            labels_placeholder = net_tensors['labels_placeholder']
            lambda_placeholder = net_tensors['lambda_placeholder']
            net_train_op = net_tensors['net_train_op']
            net_total_loss = net_tensors['net_total_loss']
            logits = net_tensors['logits']
            net_loss = net_tensors['net_loss']
            input_dict = {lambda_placeholder: lamb}

            for step in xrange(self.net_train_maxstep):
                image_datas, image_labels = self.fetch_datas(self.train_que)
                input_dict.update({images_placeholder: image_datas,
                                   labels_placeholder: image_labels})

                _, total_loss_value = sess.run([net_train_op, net_total_loss], feed_dict=input_dict)
                print('step is %d, total_loss is %f' % (step, total_loss_value))
            return total_loss_value

    def init_net_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            net_tensors = self.net_tensors
            net_init_op = net_tensors['net_init_op']

            sess.run(net_init_op)
            self.train_net(np.array([0.5] * self.batch_size))

    def train_process(self):
        lambs = self.compute_lambda()
        self.train_net(lambda_value)

    def end_net_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            net_tensors = self.net_tensors

            post_images_placeholder = net_tensors['post_images_placeholder']
            nega_images_placeholder = net_tensors['nega_images_placeholder']
            lambda_placeholder = net_tensors['lambda_placeholder']

            input_dict = {lambda_placeholder: np.array(0.5)}
            train_post_images, _ = self.distorted_inputs(train=True, post=True)
            train_nega_images, _ = self.distorted_inputs(train=True, post=False)
            input_dict.update({post_images_placeholder: train_post_images,
                                   nega_images_placeholder: train_nega_images})

            net_summary_op = net_tensors['feautre_summary_op']
            net_summary_writer = net_tensors['net_summary_writer']
            net_saver = net_tensors['net_saver']

            net_summary_str = sess.run(net_summary_op, input_dict)
            net_summary_writer.add_summary(net_summary_str)
            checkpoint_path = os.path.join(self.train_dir, 'information_pursue_%d_model.ckpt' % self.dataset_percent)
            net_saver.save(sess, checkpoint_path)


class LambNet(object):
    def __init__(self, logits, labels):
        self.logits = logits
        self.batch_size = self.logits.shape[0]
        self.devices = ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']
        self.net_device = self.devices[2]
        self.net_tensors = dict()
        self.net_graph = tf.Graph()
        self.net_sess = tf.Session(graph=self.net_graph)

    def build_network(self):
        with tf.name_scope('lamb_net'):
            logits = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 227, 227, 3))
            labels = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,))

            # put a sigfunction on logits and then transpose
            logits = tf.transpose(framwork.sig_func(logits))
            # according to the labels, erase rows which is not in labels
            labels_unique, _ = tf.unique(labels)
            labels_num = tf.size(labels_unique)
            logits = tf.gather(logits, indices=labels_unique)
            lambs = tf.gather(lambs, indices=labels_unique)
            # set the value of each row to True when it occurs in labels
            templete = tf.tile(tf.expand_dims(labels_unique, dim=1), [1, self.batch_size])
            labels_expand = tf.tile(tf.expand_dims(labels, dim=0), [labels_num, 1])
            indict_logic = tf.equal(labels_expand, templete)
            # split the tensor along rows
            logit_list = tf.split(0, labels_num, templete)
            indict_logic_list = tf.split(0, labels_num, indict_logic)
            lambda_list = tf.split(0, self.image_classes, lambs)

            loss_list = map(framwork.loss_func(), logit_list, indict_logic_list, lambda_list)
            losses = tf.add_n(loss_list)
            tensors_dict = {'labels_unique': labels_unique, 'templete': templete, 'logits_sig_trans': logits,
                            'net_losses': losses, 'indict_logic': indict_logic}
        return tensors_dict















