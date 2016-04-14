import numpy as np
import os
import random
from skimage.io import imread
from skimage.transform import resize
from threading import Thread
import Queue
import h5py


class Reader(Thread):
    def __init__(self, que, hdf5_path, hdf5_size):
        Thread.__init__(self)
        self.que = que
        self.hdf5_path = hdf5_path
        self.hdf5_size = hdf5_size
        self.setDaemon(True)

    def run(self):
        h5f = h5py.File(self.hdf5_path, 'r')
        while True:
            for index in xrange(self.hdf5_size):
                data = {}
                data['image_data'] = h5f['data_%s' % str(index)][:]
                data['image_label'] = int(h5f['label_%s' % str(index)].value)
                self.que.put(data)
                # print('index %s is putted' % str(index))


class Hdf5(object):
    def __init__(self, images_percent, images_path, work_path, image_shape, image_classes):
        self.image_dir = images_path
        self.work_path = work_path
        self.images_percent = images_percent
        self.image_shape = image_shape
        self.image_classes = image_classes
        self.train_hdf5_path = os.path.join(self.work_path, 'train_data_%s.h5' % str(images_percent))
        self.test_hdf5_path = os.path.join(self.work_path, 'test_data_%s.h5' % str(images_percent))
        self.train_hdf5_size, self.test_hdf5_size = self.hdf5_size()

    def hdf5_exist(self):
        return os.path.exists(self.train_hdf5_path) and os.path.exists(self.test_hdf5_path)

    def hdf5_size(self):
        train_image_tuples, test_image_tuples, _ = self.create_image_tuples()
        return len(train_image_tuples), len(test_image_tuples)

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
        image = image.astype(dtype=np.float32)
        return image

    def create_image_tuples(self):
        image_path_dic = self.create_image_path_dict()
        train_image_tuples = list()
        test_image_tuples = list()
        other_image_tuples = list()
        for class_label, image_paths in image_path_dic.iteritems():
            train_nums = int(len(image_paths) * self.images_percent)
            test_percent = (1 - self.images_percent) if self.images_percent >= 0.8 else 0.2
            test_nums = int(len(image_paths) * test_percent)
            train_image_paths = random.sample(image_paths, train_nums)
            test_image_paths = [image_path for image_path in image_paths if image_path not in train_image_paths]
            test_image_paths = random.sample(test_image_paths, test_nums)
            other_image_paths = [image_path for image_path in image_paths if image_path not in test_image_paths]
            train_image_tuples.extend([(train_image_path, class_label - 1) for train_image_path in train_image_paths])
            test_image_tuples.extend([(test_image_path, class_label - 1) for test_image_path in test_image_paths])
            other_image_tuples.extend([(other_image_path, class_label - 1) for other_image_path in other_image_paths])
        random.shuffle(train_image_tuples)
        random.shuffle(test_image_tuples)
        random.shuffle(other_image_tuples)
        return train_image_tuples, test_image_tuples, other_image_tuples

    def create_hdf5(self):
        if not os.path.exists(self.work_path):
            os.mkdir(self.work_path)
        train_image_tuples, test_image_tuples, other_image_tuples = self.create_image_tuples()
        image_data = dict()
        train_h5f = h5py.File(self.train_hdf5_path, 'w')
        test_h5f = h5py.File(self.test_hdf5_path, 'w')
        for index, (image_path, class_label) in enumerate(train_image_tuples):
            data = self.extract_image(image_path)
            while True:
                if data is None:
                    image_tuple = random.sample(other_image_tuples, 1)[0]
                    data = self.extract_image(image_tuple[0])
                    class_label = image_tuple[1]
                else:
                    break
            train_h5f.create_dataset('data_%s' % str(index), data=data)
            train_h5f.create_dataset('label_%s' % str(index), data=np.array(class_label))
            print('writing train %s: %s' % (str(index), image_path))

        for index, (image_path, class_label) in enumerate(test_image_tuples):
            data = self.extract_image(image_path)
            while True:
                if data is None:
                    image_tuple = random.sample(other_image_tuples, 1)[0]
                    data = self.extract_image(image_tuple[0])
                    class_label = image_tuple[1]
                else:
                    break
            test_h5f.create_dataset('data_%s' % str(index), data=data)
            test_h5f.create_dataset('label_%s' % str(index), data=np.array(class_label))
            print('writing test %s: %s' % (str(index), image_path))

        train_h5f.close()
        test_h5f.close()

    def work(self):
        if not self.hdf5_exist():
            self.create_hdf5()


class Dataset(object):
    def __init__(self, images_path, work_path, batch_size, images_percent, image_shape, image_classes):
        self.images_path = images_path
        self.work_path = work_path
        self.images_percent = images_percent
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_classes = image_classes
        self.hdf5 = Hdf5(images_percent=images_percent, images_path=images_path,
                         work_path=work_path, image_shape=image_shape, image_classes=image_classes)
        self.train_que = Queue.Queue(maxsize=300)
        self.test_que = Queue.Queue(maxsize=300)
        self.train_reader = Reader(self.train_que, self.hdf5.train_hdf5_path, self.hdf5.train_hdf5_size)
        self.test_reader = Reader(self.test_que, self.hdf5.test_hdf5_path, self.hdf5.test_hdf5_size)

    def work(self):
        self.hdf5.work()
        self.train_reader.start()
        self.test_reader.start()


















































