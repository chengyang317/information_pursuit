# coding=utf-8
__author__ = "Philip_Cheng"

import os
import re
import tensorflow as tf
import time
import information_pursue_input
import numpy as np
import datetime
import framwork
import random
import glob
from skimage.io import imread
from skimage.transform import resize

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.

tf.app.flags.DEFINE_integer('image_rows', 272, """The image's rows to be cropped.""")
tf.app.flags.DEFINE_integer('image_cols', 272, """The image's cols to be cropped.""")
tf.app.flags.DEFINE_integer('image_depth', 3, """The image's depth to be cropped.""")

# Global constants describing the CIFAR-10 data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = information_pursue_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = information_pursue_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


class Information_pursue_dataset(object):
    def __init__(self, images_path, data_dir, image_shape, dataset_percent, batch_size):
        self.images_path = images_path
        self.image_shape = image_shape
        self.dataset_dir = data_dir
        self.image_rows = image_shape[0]
        self.image_cols = image_shape[1]
        self.image_depth = image_shape[2]
        if not dataset_percent:
            raise ValueError("dataset_percent value is None")
        self.batch_size = batch_size
        self.dataset_percent = dataset_percent
        self.images_distorted_index = dict()
        self.images_records = dict()
        self.dataset = dict()
        self.dataset_create()

    def dataset_create(self):
        dataset = {}
        for path in ["253.faces-easy-101", "257.clutter"]:
            images_dir = os.path.join(self.images_path, path)
            post = "post" if path == "253.faces-easy-101" else "nega"
            train_dataset_file = os.path.join(self.dataset_dir, "train_%s_images.npy" % post)
            valid_dataset_file = os.path.join(self.dataset_dir, "valid_%s_images.npy" % post)
            if os.path.exists(train_dataset_file) and os.path.exists(valid_dataset_file):
                train_images = np.load(train_dataset_file)
                valid_images = np.load(valid_dataset_file)
            else:
                train_images, valid_images = self.extract_images(images_dir, self.dataset_percent)
                np.save(train_dataset_file, train_images)
                np.save(valid_dataset_file, valid_images)
            dataset.update({"train_%s_images" % post: train_images, "valid_%s_images" % post: valid_images})
        self.dataset = dataset

    def init_records(self, train, post):
        train_str = "train" if train else "valid"
        post_str = "post" if post else "nega"
        key = "%s_%s_images" % (train_str, post_str)
        dataset = self.dataset[key]
        images_num = dataset.shape[0]
        sequence = range(images_num)
        random.shuffle(sequence)
        self.images_distorted_index[key] = sequence
        if self.batch_size > images_num:
            raise ValueError("batch_size oversize images num")
        self.images_records[key] = 0

    def next_batch(self, train, post):
        train_str = "train" if train else "valid"
        post_str = "post" if post else "nega"
        key = "%s_%s_images" % (train_str, post_str)
        dataset = self.dataset[key]
        images_num = dataset.shape[0]
        initial_states = self.images_records.get(key, None)
        if not initial_states:
            self.init_records(train, post)
        if self.images_records[key] + self.batch_size > images_num:
            self.init_records(train, post)
        records = self.images_records[key]
        start = records
        end = records + self.batch_size
        self.images_records[key] = end
        sequence = self.images_distorted_index[key]
        images_index = sequence[start:end]
        images = dataset[images_index]
        labels = np.ones(self.batch_size, np.int32) if post else np.ones(self.batch_size, np.int32)
        return images, labels

    def generate_data(self, data, images_list):
        if data.shape[0] != len(images_list):
            raise IndexError("lenth of data is not compatible to lenth of images_list")
        for ind, image_filename in enumerate(images_list):
            image = imread(image_filename)
            if image.shape[2] != self.image_depth:
                raise ValueError("The channel of image %s is %d" % (image_filename, image.shape[2]))
            image = resize(image, self.image_shape)
            data[ind, :] = image.astype(dtype=np.float32)

    def extract_images(self, images_dir, percent):
        percent = int(percent) / float(100)
        print("Extracting %f portion images from directory %s" % (percent, images_dir))
        image_filenames = glob.glob(os.path.join(images_dir, '*.jpg'))
        nums = len(image_filenames)
        train_nums = int(nums * percent)
        valid_nums = nums - train_nums
        train_data_shape = (train_nums,) + self.image_shape
        valid_data_shape = (valid_nums,) + self.image_shape
        train_data = np.empty(train_data_shape, dtype=np.float32)
        valid_data = np.empty(valid_data_shape, dtype=np.float32)
        train_image_random_names = random.sample(image_filenames, train_nums)
        valid_image_random_names = [item for item in image_filenames if item not in train_image_random_names]
        self.generate_data(train_data, train_image_random_names)
        self.generate_data(valid_data, valid_image_random_names)
        return train_data, valid_data


class Information_pursue():
    def __init__(self, dataset_percent=20, data_dir=None, images_path=None, model_nums=5,
                 train_dir='./', batch_size=20):
        self.feature_nums = model_nums
        self.dataset_percent = dataset_percent
        self.images_path = images_path
        self.data_dir = data_dir
        self.image_shape = (227, 227, 3)
        self.learning_rate = 1e-1
        self.feature_train_maxstep = 2
        self.logits_mean_iteration = 5
        self.post_logits_batch_nums = 5
        self.nega_logits_batch_nums = 10
        self.network_percent = 0.5
        self.saturation = 1
        self.wd = 0.1

        self.log_device_placement = True
        self.devices = ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']
        self.lambda_device = self.devices[-2]
        self.feature_device = self.devices[-1]

        self.lambda_tensors = dict()
        self.feature_tensors = dict()

        self.train_dir = train_dir
        self.iteration_max_steps = 1
        self.num_classes = 2
        self.batch_size = batch_size
        self.dataset = Information_pursue_dataset(self.images_path, self.data_dir, self.image_shape,
                                                  self.dataset_percent, self.batch_size)
        self.feature_graph = tf.Graph()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=self.log_device_placement)
        config.gpu_options.allocator_type = 'BFC'
        self.feature_sess = tf.Session(graph=self.feature_graph, config=config)

    def distorted_inputs(self, train, post):
        return self.dataset.next_batch(train, post)

    def images_placeholder_input(self, train=True, post=True):
        image_shape = self.image_shape
        shape = (self.batch_size, image_shape[0], image_shape[1], image_shape[2])
        images_placeholder = tf.placeholder(tf.float32, shape=shape)
        labels_placeholder = tf.placeholder(tf.int32, shape=shape[0])
        return images_placeholder, labels_placeholder

    def logits_placeholder_input(self, batch_nums):
        shape = (self.batch_size * batch_nums, 2)
        logits_placeholder = tf.placeholder(tf.float32, shape=shape, name='logits_placeholder')
        return logits_placeholder

    def feature_placeholders(self):
        with tf.name_scope('feature_placeholders'):
            post_images_placeholder, post_labels_placeholder = self.images_placeholder_input(train=True, post=True)
            nega_images_placeholder, nega_labels_placeholder = self.images_placeholder_input(train=True, post=False)
            lambda_placeholder = tf.placeholder(dtype=tf.float32, name='lambda_placeholder')
            tensors_dict = {'post_images_placeholder': post_images_placeholder,
                            'post_labels_placeholder': post_labels_placeholder,
                            'nega_images_placeholder': nega_images_placeholder,
                            'nega_labels_placeholder': nega_labels_placeholder,
                            'lambda_placeholder': lambda_placeholder}
            return tensors_dict

    # def alex_inference(self, images, dropout_placeholder):
    #     tensors_dict = {}
    #     # first layer
    #     with tf.variable_scope("conv1") as scope:
    #         kernel, _ = framwork.variable_with_weight_decay(name='weights', shape=[11, 11, 3, 96], stddev=1e-2)
    #         conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # norm1
    #         norm = tf.nn.local_response_normalization(relu, depth_radius=5,
    #                                                   bias=1.0, alpha=1e-4, beta=0.75, name='norm1')
    #         # pool1
    #         pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    #
    #         tensors_dict.update({'conv1_kernel': kernel, 'conv1_conv': conv, 'conv1_relu': relu,
    #                              'conv1_norm': norm, 'conv1_pool': pool})
    #     # second layer
    #     with tf.variable_scope("conv2") as scope:
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[5, 5, 96, 256], stddev=1e-2)
    #         conv = tf.nn.conv2d(pool, kernel, [1, 2, 2, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # norm2
    #         norm = tf.nn.local_response_normalization(relu, depth_radius=5,
    #                                                   bias=1.0, alpha=1e-4, beta=0.75, name='norm2')
    #         # pool2
    #         pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #
    #         tensors_dict.update({'conv2_kernel': kernel, 'conv2_conv': conv, 'conv2_relu': relu,
    #                              'conv2_norm': norm, 'conv2_pool': pool})
    #     # third layer
    #     with tf.variable_scope("conv3") as scope:
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, 256, 384], stddev=1e-2)
    #         conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         tensors_dict.update({'conv3_kernel': kernel, 'conv3_conv': conv, 'conv3_relu': relu})
    #     # fourth layer
    #     with tf.variable_scope("conv4") as scope:
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, 384, 384], stddev=1e-2)
    #         conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(conv)
    #         tensors_dict.update({'conv4_kernel': kernel, 'conv4_conv': conv, 'conv4_relu': relu})
    #
    #     # fifth layer
    #     with tf.variable_scope("conv5") as scope:
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, 384, 256], stddev=1e-2)
    #         conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # pool5
    #         pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    #         tensors_dict.update({'conv5_kernel': kernel, 'conv5_conv': conv, 'conv5_relu': relu, 'conv5_pool': pool})
    #     # fully connected layer6
    #     with tf.variable_scope('fully_connected6') as scope:
    #         pool_shape = pool.get_shape().as_list()
    #         pool_ndims = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #         kernel_shape = [pool_ndims, 4096]
    #         kernel, _ = framwork.variable_with_weight_decay('weights', shape=kernel_shape, stddev=0.005)
    #         bias = framwork.variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
    #         pool = tf.reshape(pool, (pool_shape[0], pool_ndims))
    #         fc = tf.add(tf.matmul(pool, kernel), bias)
    #         relu = tf.nn.relu(fc, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # dropout
    #         drop = tf.nn.dropout(relu, dropout_placeholder)
    #         tensors_dict.update({'fc6_kernel': kernel, 'fc6_fc': fc, 'fc6_relu': relu, 'fc6_drop': drop})
    #     # fully connected layer7
    #     with tf.variable_scope('fully_connected7') as scope:
    #         kernel, _ = framwork.variable_with_weight_decay('weights', shape=[4096, 4096], stddev=0.005)
    #         bias = framwork.variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
    #         fc = tf.add(tf.matmul(drop, kernel), bias)
    #         relu = tf.nn.relu(fc, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # dropout
    #         drop = tf.nn.dropout(relu, dropout_placeholder)
    #         tensors_dict.update({'fc7_kernel': kernel, 'fc7_fc': fc, 'fc7_relu': relu, 'fc7_drop': drop})
    #     # softmax, i.e. softmax(WX + b)
    #     with tf.variable_scope('softmax_linear') as scope:
    #         weights, _ = framwork.variable_with_weight_decay('weights', [4096, 2], stddev=1e-2)
    #         biases = framwork.variable_on_cpu('biases', [2], tf.constant_initializer(0.0))
    #         softmax_linear = tf.add(tf.matmul(drop, weights), biases, name=scope.name)
    #         framwork.activation_summary(softmax_linear)
    #         tensors_dict.update({'softmax_kernel': weights, 'softmax_logits': softmax_linear})
    #     return tensors_dict
    #
    # def alex_network_percent(self, images, percent):
    #     tensors_dict = {}
    #     # first layer
    #     with tf.variable_scope("conv1") as scope:
    #         origin_nums = 96
    #         output_nums = int(origin_nums * percent)
    #         kernel, _ = framwork.variable_with_weight_decay(name='weights', shape=[11, 11, 3, output_nums], stddev=1e-2)
    #         conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.0))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # norm1
    #         norm = tf.nn.local_response_normalization(relu, depth_radius=5,
    #                                                   bias=1.0, alpha=1e-4, beta=0.75, name='norm1')
    #         # pool1
    #         pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    #
    #         tensors_dict.update({'conv1_kernel': kernel, 'conv1_conv': conv, 'conv1_relu': relu,
    #                              'conv1_norm': norm, 'conv1_pool': pool})
    #     # second layer
    #     with tf.variable_scope("conv2") as scope:
    #         input_nums = output_nums
    #         origin_nums = 256
    #         output_nums = int(origin_nums * percent)
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[5, 5, input_nums, output_nums], stddev=1e-2)
    #         conv = tf.nn.conv2d(pool, kernel, [1, 2, 2, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # norm2
    #         norm = tf.nn.local_response_normalization(relu, depth_radius=5,
    #                                                   bias=1.0, alpha=1e-4, beta=0.75, name='norm2')
    #         # pool2
    #         pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #
    #         tensors_dict.update({'conv2_kernel': kernel, 'conv2_conv': conv, 'conv2_relu': relu,
    #                              'conv2_norm': norm, 'conv2_pool': pool})
    #     # third layer
    #     with tf.variable_scope("conv3") as scope:
    #         input_nums = output_nums
    #         origin_nums = 384
    #         output_nums = int(origin_nums * percent)
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, input_nums, output_nums], stddev=1e-2)
    #         conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.0))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         tensors_dict.update({'conv3_kernel': kernel, 'conv3_conv': conv, 'conv3_relu': relu})
    #     # fourth layer
    #     with tf.variable_scope("conv4") as scope:
    #         input_nums = output_nums
    #         origin_nums = 384
    #         output_nums = int(origin_nums * percent)
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, input_nums, output_nums], stddev=1e-2)
    #         conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(conv)
    #         tensors_dict.update({'conv4_kernel': kernel, 'conv4_conv': conv, 'conv4_relu': relu})
    #
    #     # fifth layer
    #     with tf.variable_scope("conv5") as scope:
    #         input_nums = output_nums
    #         origin_nums = 256
    #         output_nums = int(origin_nums * percent)
    #         kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, input_nums, output_nums], stddev=1e-2)
    #         conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
    #         biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
    #         bias = tf.nn.bias_add(conv, biases)
    #         relu = tf.nn.relu(bias, name=scope.name)
    #         framwork.activation_summary(relu)
    #         # pool5
    #         pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    #         tensors_dict.update({'conv5_kernel': kernel, 'conv5_conv': conv, 'conv5_relu': relu, 'conv5_pool': pool})
    #     # fully connected layer6
    #     with tf.variable_scope('fully_connected6') as scope:
    #         origin_nums = 4096
    #         output_nums = int(origin_nums * percent)
    #         pool_shape = pool.get_shape().as_list()
    #         pool_ndims = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #         kernel_shape = [pool_ndims, output_nums]
    #         kernel, _ = framwork.variable_with_weight_decay('weights', shape=kernel_shape, stddev=0.005)
    #         bias = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
    #         pool = tf.reshape(pool, (pool_shape[0], pool_ndims))
    #         fc = tf.add(tf.matmul(pool, kernel), bias)
    #         relu = tf.nn.relu(fc, name=scope.name)
    #         framwork.activation_summary(relu)
    #         tensors_dict.update({'fc6_kernel': kernel, 'fc6_fc': fc, 'fc6_relu': relu})
    #     # fully connected layer7
    #     with tf.variable_scope('fully_connected7') as scope:
    #         input_nums = output_nums
    #         origin_nums = 4096
    #         output_nums = int(origin_nums * percent)
    #         kernel, _ = framwork.variable_with_weight_decay('weights', shape=[input_nums, output_nums], stddev=0.005)
    #         bias = framwork.variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
    #         fc = tf.add(tf.matmul(relu, kernel), bias)
    #         relu = tf.nn.relu(fc, name=scope.name)
    #         framwork.activation_summary(relu)
    #         tensors_dict.update({'fc7_kernel': kernel, 'fc7_fc': fc, 'fc7_relu': relu})
    #     # softmax, i.e. softmax(WX + b)
    #     with tf.variable_scope('softmax_linear') as scope:
    #         input_nums = output_nums
    #         output_nums = self.num_classes
    #         weights, _ = framwork.variable_with_weight_decay('weights', [input_nums, output_nums], stddev=1e-2)
    #         biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.0))
    #         softmax_linear = tf.add(tf.matmul(relu, weights), biases, name=scope.name)
    #         framwork.activation_summary(softmax_linear)
    #         tensors_dict.update({'softmax_kernel': weights, 'softmax_logits': softmax_linear})
    #     return tensors_dict

    def network_inference_percent(self, images, percent):
        tensors_dict = {}
        # first layer
        with tf.variable_scope("conv1") as scope:
            origin_nums = 96
            output_nums = int(origin_nums * percent)
            kernel, _ = framwork.variable_with_weight_decay(name='weights', shape=[11, 11, 3, output_nums], stddev=1e-2)
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name=scope.name)
            framwork.activation_summary(relu)
            # norm1
            norm = tf.nn.local_response_normalization(relu, depth_radius=5,
                                                      bias=1.0, alpha=1e-4, beta=0.75, name='norm')
            # pool1
            pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

            tensors_dict.update({'conv1_kernel': kernel, 'conv1_conv': conv, 'conv1_relu': relu,
                                 'conv1_norm': norm, 'conv1_pool': pool})
        # second layer
        with tf.variable_scope("conv2") as scope:
            input_nums = output_nums
            origin_nums = 256
            output_nums = int(origin_nums * percent)
            kernel, _ = framwork.variable_with_weight_decay("weights", shape=[5, 5, input_nums, output_nums], stddev=1e-2)
            conv = tf.nn.conv2d(pool, kernel, [1, 2, 2, 1], padding='SAME')
            biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name=scope.name)
            framwork.activation_summary(relu)
            # norm2
            norm = tf.nn.local_response_normalization(relu, depth_radius=5,
                                                      bias=1.0, alpha=1e-4, beta=0.75, name='norm')
            # pool2
            pool = tf.nn.max_pool(norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

            tensors_dict.update({'conv2_kernel': kernel, 'conv2_conv': conv, 'conv2_relu': relu,
                                 'conv2_norm': norm, 'conv2_pool': pool})

        # third layer
        with tf.variable_scope("conv3") as scope:
            input_nums = output_nums
            origin_nums = 256
            output_nums = int(origin_nums * percent)
            kernel, _ = framwork.variable_with_weight_decay("weights", shape=[3, 3, input_nums, output_nums], stddev=1e-2)
            conv = tf.nn.conv2d(relu, kernel, [1, 1, 1, 1], padding='SAME')
            biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name=scope.name)
            framwork.activation_summary(relu)
            # pool
            pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
            tensors_dict.update({'conv3_kernel': kernel, 'conv3_conv': conv, 'conv3_relu': relu, 'conv3_pool': pool})
        # fully connected layer
        with tf.variable_scope('fully_connected1') as scope:
            origin_nums = 4096
            output_nums = int(origin_nums * percent)
            pool_shape = pool.get_shape().as_list()
            pool_ndims = pool_shape[1] * pool_shape[2] * pool_shape[3]
            kernel_shape = [pool_ndims, output_nums]
            kernel, _ = framwork.variable_with_weight_decay('weights', shape=kernel_shape, stddev=0.005)
            bias = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.1))
            pool = tf.reshape(pool, (pool_shape[0], pool_ndims))
            fc = tf.add(tf.matmul(pool, kernel), bias)
            relu = tf.nn.relu(fc, name=scope.name)
            framwork.activation_summary(relu)
            tensors_dict.update({'fc1_kernel': kernel, 'fc1_fc': fc, 'fc1_relu': relu})

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            input_nums = output_nums
            output_nums = self.num_classes
            weights, _ = framwork.variable_with_weight_decay('weights', [input_nums, output_nums], stddev=1e-2)
            biases = framwork.variable_on_cpu('biases', [output_nums], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(relu, weights), biases, name=scope.name)
            framwork.activation_summary(softmax_linear)
            tensors_dict.update({'softmax_kernel': weights, 'softmax_logits': softmax_linear})
        return tensors_dict

    def feature_logits(self, post_images, nega_images):
        with tf.variable_scope('feature_logits') as scope:
            network_percent = self.network_percent
            post_inference_tensors = self.network_inference_percent(post_images, network_percent)
            scope.reuse_variables()
            nega_inference_tensors = self.network_inference_percent(nega_images, network_percent)
        tensors_dict = {'post_inference_tensors': post_inference_tensors, 'nega_inference_tensors': nega_inference_tensors}
        return tensors_dict

    def feature_loss(self, lambda_placeholder, post_logits, nega_logits):
        with tf.name_scope('feature_loss'):
            post_logits = tf.slice(post_logits, [0, 1], [self.batch_size, 1])
            nega_logits = tf.slice(nega_logits, [0, 0], [self.batch_size, 1])
            post_logits_sig = self.saturation * (2 / (1 + tf.exp(-2 * post_logits / self.saturation)) - 1)
            nega_logits_sig = self.saturation * (2 / (1 + tf.exp(-2 * nega_logits / self.saturation)) - 1)
            # post_logits_sig = tf.sigmoid(post_logits) * self.saturation
            # nega_logits_sig = tf.sigmoid(nega_logits) * self.saturation
            exp_nega_logits = tf.exp(nega_logits_sig * lambda_placeholder)
            Z = tf.reduce_mean(exp_nega_logits)
            log_Z = tf.log(Z)
            temp = tf.reduce_mean(lambda_placeholder * post_logits_sig)
            feature_loss = log_Z - temp
        tensors_dict = {'post_logits': post_logits, 'nega_logits': nega_logits, 'exp_nega_logits': exp_nega_logits,
                        'Z': Z, 'log_Z': log_Z, 'temp': temp, 'feature_loss': feature_loss,
                        'post_logits_sig': post_logits_sig, 'nega_logits_sig': nega_logits_sig}
        return tensors_dict

    def feature_total_loss(self, feature_loss):
        with tf.name_scope('feature_total_loss'):
            alex_inference_tensors = self.feature_tensors['post_inference_tensors']
            tensors_dict = {}
            weight_loss = list()
            for layer_str in ['conv1', 'conv2', 'conv3', 'fc1', 'softmax']:
                kernel = alex_inference_tensors[layer_str + '_kernel']
                kernel_loss = tf.mul(tf.nn.l2_loss(kernel), self.wd, name=layer_str + '_kernel_loss')
                weight_loss.append(kernel_loss)
                tensors_dict.update({layer_str + '_kernel_loss': kernel_loss})
            weight_loss.append(feature_loss)
            feature_total_loss = tf.add_n(weight_loss, name='total_loss')
            tensors_dict.update({'feature_total_loss': feature_total_loss})
            return tensors_dict

    def feature_train(self, feature_total_loss):
        with tf.name_scope('feature_train'):
            feature_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            feature_grads = feature_optimizer.compute_gradients(feature_total_loss)
            feature_train_op = feature_optimizer.apply_gradients(feature_grads)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)
            # Add histograms for gradients.
            for grad, var in feature_grads:
                if grad:
                    tf.histogram_summary(var.op.name + '/gradients', grad)
        tensors_dict = {'feature_optimizer': feature_optimizer, 'feature_grads': feature_grads,
                        'feature_train_op': feature_train_op}
        return tensors_dict

    def feature_other(self):
        with tf.name_scope('feature_other'):
            feature_init_op = tf.initialize_all_variables()
            feature_saver = tf.train.Saver(tf.all_variables())
            feautre_summary_op = tf.merge_all_summaries()
            feature_summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=self.feature_sess.graph_def)
        tensors_dict = {'feature_init_op': feature_init_op, 'feature_saver': feature_saver,
                        'feautre_summary_op': feautre_summary_op, 'feature_summary_writer': feature_summary_writer}
        return tensors_dict

    def build_feature(self):
        with self.feature_graph.as_default(), tf.device(self.feature_device), tf.name_scope('build_feature'):
            feature_tensors = self.feature_tensors

            feature_placeholders_tensors = self.feature_placeholders()
            feature_tensors.update(feature_placeholders_tensors)

            post_images_placeholder = feature_placeholders_tensors['post_images_placeholder']
            nega_images_placeholder = feature_placeholders_tensors['nega_images_placeholder']
            feature_logits_tensors = self.feature_logits(post_images_placeholder, nega_images_placeholder)
            feature_tensors.update(feature_logits_tensors)

            lambda_placeholder = feature_placeholders_tensors['lambda_placeholder']
            post_logits = feature_logits_tensors['post_inference_tensors']['softmax_logits']
            nega_logits = feature_logits_tensors['nega_inference_tensors']['softmax_logits']
            feature_loss_tensors = self.feature_loss(lambda_placeholder, post_logits, nega_logits)
            feature_tensors.update(feature_loss_tensors)

            feature_loss = feature_loss_tensors['feature_loss']
            feature_total_loss_tensors = self.feature_total_loss(feature_loss)
            feature_tensors.update(feature_total_loss_tensors)

            feature_total_loss = feature_total_loss_tensors['feature_total_loss']
            feature_train_tensors = self.feature_train(feature_total_loss)
            feature_tensors.update(feature_train_tensors)

            feature_other_tensors = self.feature_other()
            feature_tensors.update(feature_other_tensors)

    def build_model(self):
        self.build_feature()

    def logists_batch_compute(self, train, post):
        with self.feature_graph.as_default(), tf.device(self.feature_device):
            sess = self.feature_sess
            feature_tensors = self.feature_tensors
            if post:
                logits = feature_tensors['post_inference_tensors']['softmax_logits']
                images_placeholder = feature_tensors['post_images_placeholder']
            else:
                logits = feature_tensors['nega_inference_tensors']['softmax_logits']
                images_placeholder = feature_tensors['nega_images_placeholder']
            images, _ = self.distorted_inputs(train, post)
            input_dict = {images_placeholder: images}
            logits_value = sess.run([logits], feed_dict=input_dict)
        return logits_value

    def logits_mass_compute(self, train, post):
        with self.feature_graph.as_default(), tf.device(self.feature_device):
            batch_nums = self.post_logits_batch_nums if post else self.nega_logits_batch_nums
            logits_mass = np.empty((self.batch_size * batch_nums, 2))
            for i in xrange(batch_nums):
                start = self.batch_size * i
                end = start + self.batch_size
                sequence = range(start, end)
                logits_value = self.logists_batch_compute(train, post)
                logits_mass[sequence] = logits_value
        return logits_mass

    def logits_complete_compute(self):
        with self.feature_graph.as_default(), tf.device(self.feature_device):
            pass

    def formula_left_value(self, lambda_value):
        nega_logits_mass = self.logits_mass_compute(True, False)
        nega_logits_mass = nega_logits_mass[:, 0]
        nega_logits_sig = self.saturation * (2 / (1 + np.exp(-2 * nega_logits_mass / self.saturation)) - 1)
        exp_nega_logits_sig = np.exp(lambda_value * nega_logits_sig)
        Z = np.mean(exp_nega_logits_sig)
        left = np.mean(nega_logits_sig * exp_nega_logits_sig) / Z
        return left

    def compute_lambda(self):
        with tf.device(self.lambda_device):
            post_logits_mass = self.logits_mass_compute(True, True)
            post_logits_mass = post_logits_mass[:, 1]
            post_logits_sig = self.saturation * (2 / (1 + np.exp(-2 * post_logits_mass / self.saturation)) - 1)
            post_response = np.mean(post_logits_sig)

            lambda_value = -10
            for step in xrange(200):
                lambda_value += 0.1
                left_value = self.formula_left_value(lambda_value)
                print('left value is %f' % left_value)
                if np.abs(left_value - post_response) <= 0.01:
                    return lambda_value
            raise ArithmeticError('cant compute labmda value')

    def train_feature(self, lambda_value):
        with self.feature_graph.as_default(), tf.device(self.feature_device):
            sess = self.feature_sess
            feature_tensors = self.feature_tensors

            post_images_placeholder = feature_tensors['post_images_placeholder']
            nega_images_placeholder = feature_tensors['nega_images_placeholder']
            lambda_placeholder = feature_tensors['lambda_placeholder']
            feature_train_op = feature_tensors['feature_train_op']
            feature_total_loss = feature_tensors['feature_total_loss']
            input_dict = {lambda_placeholder: lambda_value}

            post_logits = feature_tensors['post_logits']
            kernel = feature_tensors['post_inference_tensors']['conv1_kernel']
            kernel_loss = feature_tensors['conv1_kernel_loss']
            feature_loss = feature_tensors['feature_loss']
            nega_logits = feature_tensors['nega_logits']
            log_z = feature_tensors['log_Z']
            temp = feature_tensors['temp']

            feature_total_loss_value = None
            for step in xrange(self.feature_train_maxstep):
                train_post_images, _ = self.distorted_inputs(train=True, post=True)
                train_nega_images, _ = self.distorted_inputs(train=True, post=False)
                input_dict.update({post_images_placeholder: train_post_images,
                                   nega_images_placeholder: train_nega_images})

                _, feature_total_loss_value = sess.run([feature_train_op, feature_total_loss], feed_dict=input_dict)
                kernel_value, kernel_loss_value, post_logits_value, feature_loss_value, nega_logits_value = sess.run(
                    [kernel, kernel_loss, post_logits, feature_loss, nega_logits], feed_dict=input_dict)
                # log_z_value, temp_value = sess.run([log_z, temp])

                print('step is %d, weight_value is %f, post logits is %f, nega logits is %f' % (step,
                        kernel_value.max(), post_logits_value.mean(), nega_logits_value.mean()))
                # print("kernel loss is %f, loss is %f, total loss is %f, " % (kernel_loss_value,
                #         feature_loss_value, feature_total_loss_value))
            return feature_total_loss_value

    def init_feature_network(self):
        with self.feature_graph.as_default(), tf.device(self.feature_device):
            sess = self.feature_sess
            feature_tensors = self.feature_tensors
            feature_init_op = feature_tensors['feature_init_op']

            sess.run(feature_init_op)
            self.train_feature(np.array(0.5))

    def train_process(self):
        lambda_value = self.compute_lambda()
        self.train_feature(lambda_value)

    def end_feature_network(self):
        with self.feature_graph.as_default(), tf.device(self.feature_device):
            sess = self.feature_sess
            feature_tensors = self.feature_tensors

            post_images_placeholder = feature_tensors['post_images_placeholder']
            nega_images_placeholder = feature_tensors['nega_images_placeholder']
            lambda_placeholder = feature_tensors['lambda_placeholder']

            input_dict = {lambda_placeholder: np.array(0.5)}
            train_post_images, _ = self.distorted_inputs(train=True, post=True)
            train_nega_images, _ = self.distorted_inputs(train=True, post=False)
            input_dict.update({post_images_placeholder: train_post_images,
                                   nega_images_placeholder: train_nega_images})

            feature_summary_op = feature_tensors['feautre_summary_op']
            feature_summary_writer = feature_tensors['feature_summary_writer']
            feature_saver = feature_tensors['feature_saver']

            feature_summary_str = sess.run(feature_summary_op, input_dict)
            feature_summary_writer.add_summary(feature_summary_str)
            checkpoint_path = os.path.join(self.train_dir, 'information_pursue_%d_model.ckpt' % self.dataset_percent)
            feature_saver.save(sess, checkpoint_path)

    def train(self):
        self.init_feature_network()
        for step in xrange(self.iteration_max_steps):
            self.train_process()
        self.end_feature_network()

    def run(self):
        self.build_model()
        self.train()
