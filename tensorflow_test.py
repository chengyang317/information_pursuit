# coding=utf-8
__author__ = "Philip_Cheng"

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    with tf.Graph().as_default(), tf.Session() as sess:
            lambda_placeholder = tf.placeholder(dtype=tf.float32)
            nega_placeholder = tf.placeholder(dtype=tf.float32, shape=[20,1])
            exp_nega_logits = tf.exp(nega_placeholder * lambda_placeholder, name="exp_nega_logits")
            Z = tf.reduce_mean(exp_nega_logits, name="Z")
            temp = nega_placeholder * exp_nega_logits
            left = tf.reduce_mean(temp, name="temp") * 1 / Z

            nega_value = np.random.rand(20, 1)

            input_dict = {nega_placeholder: nega_value}
            lambda_init = -10
            for item in xrange(2000):
                lambda_init += 0.05
                input_dict.update({lambda_placeholder: np.array(lambda_init)})
                left_value = sess.run(left, input_dict)
                print('left value is %f' % left_value)



