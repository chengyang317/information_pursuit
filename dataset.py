# coding=utf-8
__author__ = "Philip_Cheng"

import numpy as np
import lmdb
import pickle
import os
import random
from skimage.io import imread
from skimage.transform import resize
from threading import Thread


class Reader(Thread):
    def __init__(self, que, dataset_path):
        Thread.__init__(self)
        self.que = que
        self.dataset_path = dataset_path
        self.setDaemon(True)

    def run(self):
        while True:
            env = lmdb.open(self.dataset_path)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, raw_value in cursor:
                    image_data = pickle.loads(raw_value)
                    print('key is %s' % str(key))
                    self.que.put(image_data)


class DataSet(object):
    def __init__(self, percent):
        self.image_dir = '/scratch/dataset/256_ObjectCategories'
        self.dataset_dir = '/scratch/dataset/information_pursue'
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        self.percent = percent
        self.image_shape = (227, 227, 3)
        self.train_lmdb_name = 'train_caltech_lmdb_%s' % str(percent)
        self.test_lmdb_name = 'test_caltech_lmdb_%s' % str(percent)
        self.train_lmdb_path = os.path.join(self.dataset_dir, self.train_lmdb_name)
        self.test_lmdb_path = os.path.join(self.dataset_dir, self.test_lmdb_name)
        self.set_map_size()
        self.image_path_dic = self.create_image_path_dict()

    def set_map_size(self):
        temp_dict = {'label': 5, 'data': np.ones((227, 227, 3), dtype=np.float32)}
        temp_dict_size = len(pickle.dumps(temp_dict))
        self.train_map_size = int(temp_dict_size * 256 * 1000 * self.percent)
        self.test_map_size = int(temp_dict_size * 256 * 1000 * (1 - self.percent))

    def create_image_path_dict(self):
        sub_dir_names = os.listdir(self.image_dir)
        image_path_dic = dict()
        for sub_dir_name in sub_dir_names:
            image_subdir_path = os.path.join(self.image_dir, sub_dir_name)
            if not os.path.isdir(image_subdir_path):
                continue
            class_label = int(sub_dir_name[:sub_dir_name.find('.')])
            if class_label == 257:
                continue
            image_names = os.listdir(image_subdir_path)
            image_path_dic[class_label] = [os.path.join(image_subdir_path, image_name) for image_name in image_names if image_name[-3:] == 'jpg']
        return image_path_dic

    def extract_image(self, image_path):
        image = imread(image_path)
        if len(image.shape) != 3:
            return None
        image = resize(image, self.image_shape)
        data = image.astype(dtype=np.float32)
        return data

    def create_lmdb(self):
        train_env = lmdb.open(self.train_lmdb_name, map_size=self.train_map_size)
        test_env = lmdb.open(self.test_lmdb_name, map_size=self.test_map_size)
        image_path_dic = self.image_path_dic
        train_image_tuples = list()
        test_image_tuples = list()
        for class_label, image_paths in image_path_dic.iteritems():
            train_nums = int(len(image_paths) * self.percent)
            train_image_paths = random.sample(image_paths, train_nums)
            test_image_paths = [image_path for image_path in image_paths if image_path not in train_image_paths]
            train_image_tuples.extend([(train_image_path, class_label - 1) for train_image_path in train_image_paths])
            test_image_tuples.extend([(test_image_path, class_label - 1) for test_image_path in test_image_paths])
        random.shuffle(train_image_tuples)
        random.shuffle(test_image_tuples)

        image_data = dict()
        with train_env.begin(write=True) as train_txn:
            for index, (image_path, class_label) in enumerate(train_image_tuples):
                image_data['label'] = class_label
                image_data['data'] = self.extract_image(image_path)
                str_id = '{:08}'.format(index)
                train_txn.put(str_id.encode('ascii'), pickle.dumps(image_data))
                print('writing train %s: %s' % (str_id, image_path))
        with test_env.begin(write=True) as test_txn:
            for index, (image_path, class_label) in enumerate(test_image_tuples):
                image_data['label'] = class_label
                image_data['data'] = self.extract_image(image_path)
                str_id = '{:08}'.format(index)
                test_txn.put(str_id.encode('ascii'), pickle.dumps(image_data))
                print('writing test %s: %s' % (str_id, image_path))


if __name__ == '__main__':
    data_set = DataSet(0.05)
    data_set.create_lmdb()
