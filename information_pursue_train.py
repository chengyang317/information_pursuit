# coding=utf-8
__author__ = "Philip_Cheng"


import tensorflow as tf
from tensorflow.python.platform import gfile
import information_pursue as ip

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'information_pursue_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('model_nums', 5, """Number of models to train""")
tf.app.flags.DEFINE_integer('batch_size', 28,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'information_pursue_data',
                           """Path to the information_pursue data directory.""")
tf.app.flags.DEFINE_string('images_path', "256_ObjectCategories/", "The images's path")


if __name__ == '__main__':
    if gfile.Exists(FLAGS.train_dir):
        gfile.DeleteRecursively(FLAGS.train_dir)
    gfile.MakeDirs(FLAGS.train_dir)

    if not gfile.Exists(FLAGS.data_dir):
        gfile.MakeDirs(FLAGS.data_dir)

    information_pursue_model = ip.Information_pursue(dataset_percent=20,
                data_dir=FLAGS.data_dir, images_path=FLAGS.images_path, model_nums= FLAGS.model_nums,
                train_dir = FLAGS.train_dir, batch_size=FLAGS.batch_size)
    information_pursue_model.run()