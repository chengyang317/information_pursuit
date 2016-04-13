import tensorflow as tf
import numpy as np
import framwork
import os


class InforNet(object):
    # network class for Information Pursue Model
    def __init__(self, batch_size, lamb_net, network_percent, work_path, data_set):
        self.work_path = work_path
        self.lamb_net = lamb_net
        self.network_percent = network_percent
        self.data_set = data_set
        self.image_shape = data_set.image_shape
        self.image_classes = data_set.image_classes
        self.batch_size = data_set.batch_size
        self.whole_images = None
        self.whole_labels = None
        self.check_point_name = 'ConvNetModel_%s_%s.ckpt' % (str(network_percent), str(self.data_set.images_percent))
        self.check_point_path = os.path.join(self.work_path, self.check_point_name)
        self.net_params = {'weight_decay': 0.1, 'learning_rate': 1e-1, 'train_loops': 1000,
                           'devices': ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']}
        self.net_device = self.net_params['devices'][2]
        self.net_tensors = dict()
        self.tensors_names = list()
        self.net_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        self.net_sess = tf.Session(graph=self.net_graph, config=config)

    def build_placeholders(self):
        shape = (self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        labels = tf.placeholder(dtype=tf.int32, shape=shape[0], name='labels')
        lambs = tf.placeholder(dtype=tf.float32, shape=self.image_classes, name='lambs')
        tensors_dict = {'images': images, 'labels': labels, 'lambs': lambs}
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
        tensors_dict.update(framwork.add_conv_layer(layer_name, input_image, kernel_attrs=kernel_attrs,
                                                    norm_attrs=norm_attrs, pool_attrs=pool_attrs))
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

    def build_loss(self, logits, labels, lambs):
        # put a sigfunction on logits and then transpose
        logits = tf.transpose(framwork.sig_func(logits))
        # according to the labels, erase rows which is not in labels
        labels_unique = tf.constant(range(self.image_classes), dtype=tf.int32)
        labels_num = self.image_classes
        logits = tf.gather(logits, indices=labels_unique)
        lambs = tf.gather(lambs, indices=labels_unique)
        # set the value of each row to True when it occurs in labels
        template = tf.tile(tf.expand_dims(labels_unique, dim=1), [1, self.batch_size])
        labels_expand = tf.tile(tf.expand_dims(labels, dim=0), [labels_num, 1])
        indict_logic = tf.equal(labels_expand, template)
        # split the tensor along rows
        logit_list = tf.split(0, labels_num, logits)
        indict_logic_list = tf.split(0, labels_num, indict_logic)
        lambda_list = tf.split(0, self.image_classes, lambs)
        # loss_list = list()
        # for i in range(self.image_classes):
        #     loss_list.append(framwork.loss_func(logit_list[i], indict_logic_list[i], lambda_list[i]))
        loss_list = map(framwork.loss_func, logit_list, indict_logic_list, lambda_list)
        loss = tf.add_n(loss_list)
        tensors_dict = {'labels_unique': labels_unique, 'template': template, 'logits_sig_trans': logits,
                        'loss': loss, 'indict_logic': indict_logic}
        self.tensors_names.extend(tensors_dict.keys())
        self.net_tensors.update(tensors_dict)

    def build_total_loss(self):
        weight_loss = list()
        tensors_dict = {}
        for layer_name in ['conv1', 'conv2', 'conv3', 'full', 'softmax']:
            kernel = self.net_tensors[layer_name + '_kernel']
            kernel_loss = tf.mul(tf.nn.l2_loss(kernel), self.net_params['weight_decay'],
                                 name=layer_name + '_kernel_loss')
            weight_loss.append(kernel_loss)
            tensors_dict.update({layer_name + '_kernel_loss': kernel_loss})
        weight_loss.append(self.net_tensors['loss'])
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
            self.build_loss(self.net_tensors['logits'], self.net_tensors['labels'], self.net_tensors['lambs'])
            self.build_total_loss()
            self.build_eval(self.net_tensors['logits'], self.net_tensors['labels'])
            self.build_train(self.net_tensors['total_loss'])
            self.build_other()

    def fetch_batch_data(self, que):
        image_datas = list()
        for i in xrange(self.batch_size):
            image_data = que.get()
            image_datas.append(image_data)
        que.task_done()
        images = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)
        labels = np.empty(self.batch_size, dtype=np.int32)
        for i in xrange(self.batch_size):
            images[i, :] = image_data['image_data'].astype(dtype=np.float32)
            labels[i] = np.array(image_data['image_label'])
        return images, labels

    def train_network(self, lamb_datas):
        with self.net_graph.as_default(), tf.device(self.net_device):
            images = self.net_tensors['images']
            labels = self.net_tensors['labels']
            lambs = self.net_tensors['lambs']
            train_op = self.net_tensors['train_op']
            total_loss = self.net_tensors['total_loss']
            input_dict = {lambs: lamb_datas}

            for step in xrange(self.net_params['train_loops']):
                image_datas, image_labels = self.fetch_datas(self.data_set.train_que)
                input_dict.update({images: image_datas, labels: image_labels})
                _, total_loss_value = self.net_sess.run([train_op, total_loss], feed_dict=input_dict)
                if step % 20 == 0:
                    print('step is %d, total_loss is %f' % (step, total_loss_value))

    def init_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            net_tensors = self.net_tensors
            init_op = net_tensors['init_op']
            sess.run(init_op)
            self.train_network(np.array([0.5] * self.batch_size))

    def compute_lambs(self):
        lamb_batch_size = self.lamb_net.batch_size
        batch_num = self.batch_size / lamb_batch_size
        batch_images = list()
        batch_labels = list()
        for i in xrange(batch_num):
            images, labels = self.fetch_batch_data(self.data_set.train_que)


    def train_process(self):
        lambs = self.lamb_net.work()
        self.train_net(lambda_value)

    def end_network(self):
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
            checkpoint_path = os.path.join(self.work_path, 'information_pursue_%d_model.ckpt' % self.dataset_percent)
            net_saver.save(sess, checkpoint_path)

    def train_network(self):
        self.init_feature_network()
        for step in xrange(self.iteration_max_steps):
            self.train_process()
        self.end_feature_network()

    def run(self):
        self.train_network()


class LambNet(object):
    def __init__(self, batch_size, image_classes):
        self.batch_size = batch_size
        self.image_classes = image_classes
        self.devices = ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']
        self.net_device = self.devices[2]
        self.net_tensors = dict()
        self.net_graph = tf.Graph()
        self.net_sess = tf.Session(graph=self.net_graph)

    def build_network(self):
        net_tensors = self.net_tensors
        with self.net_graph.as_default(), tf.device(self.net_device):
            logits = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.image_classes))
            labels = tf.placeholder(dtype=tf.int32, shape=(self.batch_size,))
            lambs = tf.placeholder(dtype=tf.float32, shape=(self.image_classes,))
            # put a sigfunction on logits and then transpose
            logits = tf.transpose(framwork.sig_func(logits))
            # according to the labels, erase rows which is not in labels

            labels_unique = tf.constant(range(self.image_classes), dtype=tf.int32)
            labels_num = self.image_classes
            logits = tf.gather(logits, indices=labels_unique)
            lambs = tf.gather(lambs, indices=labels_unique)
            # set the value of each row to True when it occurs in labels
            templete = tf.tile(tf.expand_dims(labels_unique, dim=1), [1, self.batch_size])
            labels_expand = tf.tile(tf.expand_dims(labels, dim=0), [labels_num, 1])
            indict_logic = tf.equal(labels_expand, templete)
            # split the tensor along rows
            logit_list = tf.split(0, labels_num, logits)
            indict_logic_list = tf.split(0, labels_num, indict_logic)
            lamb_list = tf.split(0, self.image_classes, lambs)
            logit_list = [tf.squeeze(item) for item in logit_list]
            indict_logic_list = [tf.squeeze(item) for item in indict_logic_list]
            left_right_tuples = list()
            for i in range(self.image_classes):
                left_right_tuples.append(framwork.lamb_func(logit_list[i], indict_logic_list[i], lamb=lamb_list[i]))
            # func = framwork.lamb_func()
            # left_right_tuples = map(func, logit_list, indict_logic_list, lamb_list)
            net_tensors.update({'left_right_tuples': left_right_tuples, 'logits': logits, 'labels': labels,
                                'lambs': lambs})

    def compute_lambs(self, logits, labels):
        with self.net_graph.as_default(), tf.device(self.net_device):
            net_tensors = self.net_tensors
            lambs = [1.0] * self.image_classes
            input_dict = {net_tensors['logits']: logits, net_tensors['labels']: labels,
                          net_tensors['lambs']: np.array(lambs)}
            sess = self.net_sess
            left_right_tuples = net_tensors['left_right_tuples']
            left_rights = list()

            states = [None] * self.image_classes
            steps = [0.1] * self.image_classes
            moment_reg = 1.3
            moment_rev = 0.5
            map(lambda tup: left_rights.extend([tup[0], tup[1]]), left_right_tuples)
            for i in range(20):
                left_right_values = sess.run(left_rights, feed_dict=input_dict)
                for i in range(self.image_classes):
                    left = left_right_values[i]
                    right = left_right_values[i + 1]
                    lamb = lambs[i]
                    state = states[i]
                    step = steps[i]
                    state_now = False if left > right else True
                    if state == None:
                        state = state_now
                        step = step * moment_reg
                        step_vec = step * moment_reg if state_now else (- step * moment_reg)
                        lamb = lamb + step_vec
                    elif state:
                        state = state_now
                        step = step * moment_reg if state_now else step * moment_rev
                        step_vec = step if state_now else (- step)
                        lamb = lamb + step_vec
                    else:
                        state = state_now
                        step = step * moment_rev if state_now else step * moment_reg
                        step_vec = (- step) if state_now else step
                        lamb = lamb + step_vec
            return lambs

    def work(self):
        self.build_network()


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
        self.net_params = {'weight_decay': 0.1, 'learning_rate': 1e-1, 'train_loops': 1000,
                           'devices': ['/cpu:0', '/gpu:0', '/gpu:1', '/gpu:2']}
        self.net_device = self.net_params['devices'][3]
        self.net_tensors = dict()
        self.tensors_names = list()
        self.net_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
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
            image_data = que.get()
            que.task_done()
            image_datas.append(image_data)
        images = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)
        labels = np.empty(self.batch_size, dtype=np.int32)

        for i in xrange(self.batch_size):
            images[i, :] = image_data['image_data'].astype(dtype=np.float32)
            labels[i] = np.array(image_data['image_label'])
        return images, labels

    def train_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            images = self.net_tensors['images']
            labels = self.net_tensors['labels']
            train_op = self.net_tensors['train_op']
            total_loss = self.net_tensors['total_loss']
            # logits = self.net_tensors['logits']
            # loss = self.net_tensors['cross_entropy_mean']
            input_dict = {}
            for step in xrange(self.net_params['train_loops']):
                image_datas, image_labels = self.fetch_batch_datas(self.data_set.train_que)
                input_dict.update({images: image_datas, labels: image_labels})
                _, total_loss_value = sess.run([train_op, total_loss], feed_dict=input_dict)
                if step % 20 == 0:
                    print('step is %d, total_loss is %f' % (step, total_loss_value))
                if step % 100 == 0 and step != 0:
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
        for i in xrange(batch_num):
            predictions = self.net_sess.run([self.net_tensors['top_k_op']])
            true_count += np.sum(predictions)
        precision = float(true_count) / total_count
        print('precision is %f' % precision)

    def work(self):
        self.build_network()
        self.init_network()
        self.train_network()
