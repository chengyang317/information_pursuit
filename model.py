import net
import tensorflow as tf
import dataset
import Queue
import os


class InforModel(object):

    def __init__(self):
        self.train_path = '/scratch/dataset/information_pursue'
        self.dataset_path = '/scratch/dataset/information_pursue'
        self.images_path = '/scratch/dataset/256_ObjectCategories'
        self.dataset_percent = 0.05
        self.network_percent = 0.5
        self.batch_size = 100
        self.lamb_batch_num = 10
        self.dataset = dataset.DataSet(percent=self.dataset_percent, images_path=self.images_path,
                                       dataset_path=self.dataset_path)
        self.create_dataset()
        self.train_que = Queue.Queue(maxsize=1500)
        self.test_que = Queue.Queue(maxsize=1500)
        self.train_reader = dataset.Reader(self.train_que, self.dataset.train_lmdb_path)
        self.test_reader = dataset.Reader(self.test_que, self.dataset.test_lmdb_path)

        self.lamb_net = net.LambNet(batch_size=self.batch_size * self.lamb_batch_num)
        self.infor_net = net.InforNet(network_percent=self.network_percent, train_path=self.train_path,
                                      train_que=self.train_que, test_que=self.test_que, batch_size=self.batch_size,
                                      lamb_net=self.lamb_net)

    def create_dataset(self):
        if not self.dataset.lmdb_exist():
            self.dataset.create_lmdb()

    def run(self):
        self.infor_net.run()


if __name__ == '__main__':
    infor_model = InforModel()
    infor_model.run()













