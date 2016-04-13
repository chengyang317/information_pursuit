import net
import tensorflow as tf
import dataset
import Queue
import os


class InforModel(object):
    def __init__(self):
        self.work_path = '/scratch/dataset/information_pursue'
        self.dataset_path = '/scratch/dataset/information_pursue'
        self.images_path = '/scratch/dataset/256_ObjectCategories'
        self.images_percent = 0.05
        self.network_percent = 0.5
        self.batch_size = 100
        self.image_shape = (227, 227, 3)
        self.image_classes = 256
        self.dataset = dataset.Dataset(images_path=self.images_path, work_path=self.work_path,
                                       batch_size=self.batch_size, images_percent=self.images_percent,
                                       image_shape= self.image_shape, image_classes=self.image_classes)
        self.lamb_batch_num = self.dataset.lmdb.train_lmdb_size / self.batch_size
        self.lamb_net = net.LambNet(batch_size=self.batch_size * self.lamb_batch_num)
        self.infor_net = net.InforNet(network_percent=self.network_percent, train_path=self.work_path,
                                      train_que=self.train_que, test_que=self.test_que, batch_size=self.batch_size,
                                      lamb_net=self.lamb_net)

    def work(self):
        self.infor_net.run()


class ConvModel(object):
    def __init__(self, images_percent, network_percent, batch_size):
        self.work_path = '/scratch/dataset/information_pursue'
        self.dataset_path = '/scratch/dataset/information_pursue'
        self.images_path = '/scratch/dataset/256_ObjectCategories'
        self.images_percent = images_percent
        self.network_percent = network_percent
        self.batch_size = batch_size
        self.image_shape = (227, 227, 3)
        self.image_classes = 256
        self.data_set = dataset.Dataset(images_path=self.images_path, work_path=self.work_path,
                                       batch_size=self.batch_size, images_percent=self.images_percent,
                                       image_shape= self.image_shape, image_classes=self.image_classes)
        self.conv_net = net.ConvNet(batch_size=self.batch_size, network_percent=self.network_percent,
                                    work_path=self.work_path, data_set=self.data_set)

    def work(self):
        self.data_set.work()
        self.conv_net.work()












