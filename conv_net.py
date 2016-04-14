import numpy as np
import framwork
import os
import tensorflow as tf

class ConvNet(object):
    # network class for Information Pursue Model
    def __init__(self, batch_size, network_percent, work_path, data_set):
        self.work_path = work_path
        self.network_percent = network_percent
        self.data_set = data_set
        self.image_shape = data_set.image_shape
        self.image_classes = data_set.image_classes
        self.batch_size = data_set.batch_size
        self.check_point_name = 'ConvNetModel_%s_%s.ckpt' % (str(network_percent), str(self.data_set.images_percent))
        self.check_point_path = os.path.join(self.work_path, self.check_point_name)
        self.net_params = {'weight_decay': 0.1, 'learning_rate': 0.1, 'train_loops': 10000,
                           'devices': ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']}
        self.net_device = self.net_params['devices'][3]
        self.net_tensors = dict()
        self.tensors_names = list()
        self.net_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allocator_type = 'BFC'
        self.net_sess = tf.Session(graph=self.net_graph, config=config)

    def build_placeholders(self):
        shape = (self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        images_placeholder = tf.placeholder(dtype=tf.float32, shape=shape, name='images_placeholder')
        labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(shape[0],), name='labels_placeholder')
        tensors_dict = {'images': images_placeholder, 'labels': labels_placeholder}
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_logits(self, images):
        percent = self.network_percent
        tensors_dict = dict()
        # first layer
        input_image = images
        layer_name = 'conv1'
        kernel_attrs = {'shape': [11, 11, 3, int(96 * percent)], 'stddev': 1e-2, 'strides': [1, 4, 4, 1],
                        'padding': 'SAME', 'biase': 0.0}
        norm_attrs = {'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75}
        pool_attrs = {'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'}
        tensors_dict.update(framwork.add_conv_layer(layer_name, input_image, kernel_attrs=kernel_attrs, norm_attrs=norm_attrs, pool_attrs=pool_attrs))
        # second layer
        input_image = tensors_dict[layer_name + '_pool']
        layer_name = 'conv2'
        update_kernel = {'shape': [5, 5, int(96 * percent), int(256 * percent)],
                         'strides': [1, 2, 2, 1], 'bias': 0.1}
        kernel_attrs.update(update_kernel)
        tensors_dict.update(framwork.add_conv_layer(layer_name, input_image, kernel_attrs=kernel_attrs,
                                                    norm_attrs=norm_attrs, pool_attrs=pool_attrs))
        # third layer
        input_image = tensors_dict[layer_name + '_pool']
        layer_name = 'conv3'
        update_kernel = {'shape': [3, 3, int(256 * percent), int(256 * percent)],
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
        update_kernel = {'shape': [input_image_ndims, int(4096 * percent)], 'stddev': 0.005}
        kernel_attrs.update(update_kernel)
        tensors_dict.update(framwork.add_full_layer(layer_name, input_image, kernel_attrs=kernel_attrs))
        # softmax
        input_image = tensors_dict[layer_name + '_relu']
        layer_name = 'softmax'
        update_kernel = {'shape': [int(4096 * percent), self.image_classes], 'stddev': 1e-2}
        kernel_attrs.update(update_kernel)
        tensors_dict.update(framwork.add_softmax_layer(layer_name, input_image, kernel_attrs=kernel_attrs))
        # add a alias name to logits
        tensors_dict.update({'logits': tensors_dict[layer_name + '_softmax']})
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tensors_dict = {'cross_entropy': cross_entropy, 'cross_entropy_mean': cross_entropy_mean}
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_total_loss(self):
        weight_loss = list()
        tensors_dict = {}
        for layer_name in ['conv1', 'conv2', 'conv3', 'full', 'softmax']:
            kernel = self.net_tensors[layer_name + '_kernel']
            kernel_loss = tf.mul(tf.nn.l2_loss(kernel), self.net_params['weight_decay'], name=layer_name + '_kernel_loss')
            weight_loss.append(kernel_loss)
            tensors_dict.update({layer_name + '_kernel_loss': kernel_loss})
        weight_loss.append(self.net_tensors['cross_entropy_mean'])
        total_loss = tf.add_n(weight_loss, name='total_loss')
        tensors_dict.update({'total_loss': total_loss})
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_eval(self, logits, labels):
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        tensors_dict = {'top_k_op': top_k_op}
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_train(self, total_loss):
        optimizer = tf.train.GradientDescentOptimizer(self.net_params['learning_rate'])
        grads = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads)
        tensors_dict = {'optimizer': optimizer, 'grads': grads, 'train_op': train_op}
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_other(self):
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.work_path, graph_def=self.net_sess.graph_def)
        tensors_dict = {'init_op': init_op, 'saver': saver, 'summary_op': summary_op, 'summary_writer': summary_writer}
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            self.build_placeholders()
            self.build_logits(self.net_tensors['images'])
            self.build_loss(self.net_tensors['logits'], self.net_tensors['labels'])
            self.build_total_loss()
            self.build_eval(self.net_tensors['logits'], self.net_tensors['labels'])
            self.build_train(self.net_tensors['total_loss'])
            self.build_other()

    def fetch_batch_datas(self, que):
        image_datas = list()
        for i in xrange(self.batch_size):
            image_datas.append(que.get())
            que.task_done()
        images = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)
        labels = np.empty(self.batch_size, dtype=np.int32)

        for i in xrange(self.batch_size):
            images[i, :] = image_datas[i]['image_data'].astype(dtype=np.float32)
            labels[i] = np.array(image_datas[i]['image_label'])
        return images, labels

    def train_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            images = self.net_tensors['images']
            labels = self.net_tensors['labels']
            train_op = self.net_tensors['train_op']
            loss = self.net_tensors['cross_entropy_mean']
            total_loss = self.net_tensors['total_loss']
            loss1 = self.net_tensors['conv1_kernel_loss']
            loss2 = self.net_tensors['conv2_kernel_loss']
            loss3 = self.net_tensors['conv3_kernel_loss']
            loss4 = self.net_tensors['full_kernel_loss']
            kernel4 = self.net_tensors['full_kernel']
            loss5 = self.net_tensors['softmax_kernel_loss']
            # logits = self.net_tensors['logits']
            # loss = self.net_tensors['cross_entropy_mean']
            input_dict = {}
            for step in xrange(self.net_params['train_loops']):
                image_datas, image_labels = self.fetch_batch_datas(self.data_set.train_que)
                input_dict.update({images: image_datas, labels: image_labels})
                # _, total_loss_value = sess.run([train_op, total_loss], feed_dict=input_dict)
                _, k4, l1, l2, l3, l4, l5, loss_value, total_loss_value = sess.run([train_op, kernel4, loss1, loss2, loss3, loss4, loss5, loss ,total_loss], feed_dict=input_dict)
                if step % 20 == 0:
                    print('step is %d, total_loss is %f, loss is %f' % (step, total_loss_value, loss_value))
                if step % 500 == 0 and step != 0:
                    self.eval_network()
                if step % 1000 == 0 and step != 0:
                    self.save_network()

    def init_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            init_op = self.net_tensors['init_op']
            self.net_sess.run(init_op)

    def save_network(self, step=None):
        with self.net_graph.as_default(), tf.device(self.net_device):
            if not step:
                check_point_path = self.check_point_path
            else:
                check_point_path = self.check_point_path[:-5] + '_%s.ckpt' % str(step)
            saver = self.net_tensors['saver']
            saver.save(self.net_sess, check_point_path)

    def eval_network(self):
        batch_num = self.data_set.hdf5.test_hdf5_size / self.batch_size
        true_count = 0
        total_count = batch_num * self.batch_size
        images = self.net_tensors['images']
        labels = self.net_tensors['labels']
        input_dict = {}
        for i in xrange(batch_num):
            image_datas, image_labels = self.fetch_batch_datas(self.data_set.test_que)
            input_dict.update({images: image_datas, labels: image_labels})
            predictions = self.net_sess.run([self.net_tensors['top_k_op']], feed_dict=input_dict)
            true_count += np.sum(predictions)
        precision = float(true_count) / total_count
        print('precision is %f' % precision)

    def work(self):
        self.build_network()
        self.init_network()
        self.train_network()

