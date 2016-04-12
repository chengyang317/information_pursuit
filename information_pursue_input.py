# coding=utf-8
__author__ = "Philip_Cheng"

import os
import tensorflow as tf

import numpy as np
import random
import glob
from skimage.io import imread
from skimage.transform import resize

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000


class Create_records(object):
    def __init__(self):
        pass

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_to(self, images, labels, name):
        num_examples = labels.shape[0]
        if images.shape[0] != num_examples:
            raise ValueError("Images size %d does not match label size %d." %
                            (images.shape[0], num_examples))
        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        filename = name
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': self._int64_feature(rows),
                'width': self._int64_feature(cols),
                'depth': self._int64_feature(depth),
                'label': self._int64_feature(int(labels[index])),
                'image_raw': self._bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())


class Information_records(Create_records):
    def __init__(self, data_dir, rows=227, cols=227, depth=3):
        super(Information_records, self).__init__()
        self.data_dir = data_dir
        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.new_shape = (self.rows, self.cols, self.depth)

    def generate_data(self, data, images_list):
        if data.shape[0] != len(images_list):
            raise IndexError("lenth of data is not compatible to lenth of images_list")
        for ind, image_filename in enumerate(images_list):
            image = imread(image_filename)
            if image.shape[2] != self.depth:
                raise ValueError("The channel of image %s is %d" % (image_filename, image.shape[2]))
            image = resize(image, self.new_shape)
            data[ind, :] = image.astype(dtype=np.float32)

    def extract_images(self, images_dir, percent):
        percent = int(percent)/float(100)
        print("Extracting %f portion images from directory %s" % (percent, images_dir))
        image_filenames = glob.glob(os.path.join(images_dir, '*.jpg'))
        nums = len(image_filenames)
        train_nums = int(nums * percent)
        valid_nums = nums - train_nums
        train_data_shape = (train_nums,) + self.new_shape
        valid_data_shape = (valid_nums,) + self.new_shape
        train_data = np.empty(train_data_shape, dtype=np.float32)
        valid_data = np.empty(valid_data_shape, dtype=np.float32)
        train_image_random_names = random.sample(image_filenames, train_nums)
        valid_image_random_names = [item for item in image_filenames if item not in train_image_random_names]
        self.generate_data(train_data, train_image_random_names)
        self.generate_data(valid_data, valid_image_random_names)
        return train_data, valid_data

    def extract_labels(self, images_dir, percent, postive=True):
        percent = int(percent)/float(100)
        image_filenames = glob.glob(os.path.join(images_dir, '*.jpg'))
        nums = len(image_filenames)
        train_nums = int(nums * percent)
        valid_nums = nums - train_nums
        if postive:
            train_labels = np.ones(train_nums)
            valid_labels = np.ones(valid_nums)
        else:
            train_labels = np.zeros(train_nums)
            valid_labels = np.zeros(valid_nums)
        return train_labels, valid_labels

    @staticmethod
    def get_dataset_size(images_path, percent, train=True, postive=True):
        if train:
            images_dir = os.path.join(images_path, "253.faces-easy-101")
        else:
            images_dir = os.path.join(images_path, "257.clutter")
        percent = int(percent)/float(100)
        image_filenames = glob.glob(os.path.join(images_dir, '*.jpg'))
        nums = len(image_filenames)
        train_nums = int(nums * percent)
        valid_nums = nums - train_nums
        return train_nums if postive else valid_nums

    def create_records(self, images_dir, postive, percent):
        train_images, valid_images = self.extract_images(images_dir, percent)
        train_labels, valid_labels = self.extract_labels(images_dir, percent, postive)
        records_name = "postive" if postive else "negative"
        records_name = records_name + "_%s" % percent + '.tfrecords'
        train_records_name = os.path.join(self.data_dir, "train_" + records_name)
        valid_records_name = os.path.join(self.data_dir, "valid_" + records_name)
        self.convert_to(train_images, train_labels, name=train_records_name)
        self.convert_to(valid_images, valid_labels, name=valid_records_name)

    def create_datasets(self, images_path, percent):
        self.create_records(os.path.join(images_path, "253.faces-easy-101"), True, percent)
        self.create_records(os.path.join(images_path, "257.clutter"), False, percent)


class Information_pursue_input(object):
    def __init__(self, image_rows, image_cols, image_depth, num_classes):
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.image_depth = image_depth
        self.num_classes = num_classes

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, dense_keys=['image_raw', 'label'],
                    dense_types=[tf.string, tf.int64])
        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image_shape = [self.image_rows, self.image_cols, self.image_depth]
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        # image.set_shape(image_shape)
        image = tf.reshape(image, image_shape)
        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.
        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.cast(image, tf.float32)
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        return image, label


    def _generate_image_and_label_batch(self, image, label, min_queue_examples):
        """Construct a queued batch of images and labels.
        Args:
            image: 3-D Tensor of [height, width, 3] of type.float32.
            label: 1-D Tensor of type.int32
            min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples.
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, height, width, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        num_preprocess_threads = 4
        batch_size = self.batch_size
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

        # Display the training images in the visualizer.
        tf.image_summary('images', images)
        return images, tf.reshape(label_batch, [batch_size])

    def distorted_inputs(self, dataset_filename, batch_size, num_epochs=None):
        if not os.path.isfile(dataset_filename):
            raise ValueError("file %s is not existed" % dataset_filename)
        self.dataset_filename = dataset_filename
        if not batch_size:
            raise ValueError("batch_size is None")
        self.batch_size = batch_size
        filename = self.dataset_filename
        with tf.name_scope("distorted_input") as scope:
            filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
            image, label = self.read_and_decode(filename_queue)
            # Randomly flip the image horizontally.

            # Randomly crop a [height, width] section of the image.
            # distorted_image = tf.image.random_crop(reshaped_image, [height, width])

            distorted_image = tf.image.random_flip_left_right(image)

            # Because these operations are not commutative, consider randomizing
            # randomize the order their operation.
            distorted_image = tf.image.random_brightness(distorted_image,
                                                   max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                 lower=0.2, upper=1.8)

            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_whitening(distorted_image)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                               min_fraction_of_examples_in_queue)
            print ('Filling queue with %d CIFAR images before starting to train. '
                'This will take a few minutes.' % min_queue_examples)

            # Generate a batch of images and labels by building up a queue of examples.
            return self._generate_image_and_label_batch(float_image, label,
                                             min_queue_examples)

