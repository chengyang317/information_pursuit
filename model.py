import net
import tensorflow as tf
import dataset


class InforModel(object):

    def __init__(self):
        self.train_path = '/scratch/dataset/information_pursue'
        self.dataset_path = '/scratch/dataset/information_pursue'
        self.images_path = '/scratch/dataset/256_ObjectCategories'
        self.dataset_percent = 0.05
        self.network_percent = 0.5
        self.batch_size = 100
        self.lamb_batch_num = 10
        self.dataset = dataset.DataSet(percent=self.dataset_percent, images_path=self.images_path, dataset_path=self.dataset_path)
        self.infor_net = net.InforNet(network_percent=self.network_percent, train_path=self.train_path,
                                      batch_size=self.batch_size, lamb_batch_num=self.lamb_batch_num)
        self.lamb_net = net.LambNet(batch_size=self.batch_size * self.lamb_batch_num)

    def create_dataset(self):
        if not self.dataset.lmdb_exist():
            self.dataset.create_lmdb()















