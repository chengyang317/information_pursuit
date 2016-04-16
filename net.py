import numpy as np
import os
import tensorflow as tf


class Net(object):
    # network class for Information Pursue Model
    def __init__(self, net_name, batch_size, net_device, net_percent, work_path, data_set):
        self.net_name = net_name
        self.work_path = work_path
        self.net_percent = net_percent
        self.data_set = data_set
        self.image_shape = data_set.image_shape
        self.image_classes = data_set.image_classes
        self.batch_size = batch_size
        self.net_device = net_device
        self.check_point_name = '%s_%s_%s.ckpt' % (net_name, str(net_percent), str(self.data_set.images_percent))
        self.check_point_path = os.path.join(self.work_path, self.check_point_name)
        self.net_tensors = dict()
        self.loss_list = list()
        self.net_graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allocator_type = 'BFC'
        self.net_sess = tf.Session(graph=self.net_graph, config=config)

    def fetch_batch_datas(self, que):
        datas = list()
        for i in xrange(self.batch_size):
            datas.append(que.get())
            que.task_done()
        images = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)
        labels = np.empty(self.batch_size, dtype=np.int32)
        for i in xrange(self.batch_size):
            images[i, :] = datas[i]['image_data'].astype(dtype=np.float32)
            labels[i] = np.array(datas[i]['image_label'])
        return images, labels

    def build_network(self):
        pass

    def train_network(self):
        pass

    def init_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            init_op = self.net_tensors['aux_init_op']
            self.net_sess.run(init_op)

    def save_network(self, step=None):
        with self.net_graph.as_default(), tf.device(self.net_device):
            if not step:
                check_point_path = self.check_point_path
            else:
                check_point_path = self.check_point_path[:-5] + '_%s.ckpt' % str(step)
            saver = self.net_tensors['aux_saver']
            saver.save(self.net_sess, check_point_path)

    def eval_network(self):
        batch_num = self.data_set.hdf5.test_hdf5_size / self.batch_size
        true_count = 0
        total_count = batch_num * self.batch_size
        images = self.net_tensors['input_images']
        labels = self.net_tensors['input_labels']
        input_dict = {}
        for i in xrange(batch_num):
            image_datas, image_labels = self.fetch_batch_datas(self.data_set.test_que)
            input_dict.update({images: image_datas, labels: image_labels})
            predictions = self.net_sess.run([self.net_tensors['eval_op']], feed_dict=input_dict)
            true_count += np.sum(predictions)
        precision = float(true_count) / total_count
        print('precision is %f' % precision)

    def work(self):
        self.build_network()
        self.init_network()
        self.train_network()

