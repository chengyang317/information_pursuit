import os
import numpy as np

caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe



def calculate_mean(txt_file, out_path):

    crop_dim = (227, 227)
    with open(txt_file) as txt_fd:
        lines_list = txt_fd.readlines()
    face_jpgs_list = [line.strip().split()[0] for line in lines_list if line.strip().split()[1] is '1']
    cluster_jpgs_list = [line.strip().split()[0] for line in lines_list if line.strip().split()[1] is '0']
    face_array = np.empty((len(face_jpgs_list),) + crop_dim + (3, ), dtype=float)
    cluster_array = np.empty((len(cluster_jpgs_list),) + crop_dim + (3, ), dtype=float)
    for ind, jpg in enumerate(face_jpgs_list):
        image = caffe.io.load_image(jpg)
        image = caffe.io.resize_image(image, crop_dim)
        face_array[ind, :] = image
    for ind, jpg in enumerate(cluster_jpgs_list):
        image = caffe.io.load_image(jpg)
        image = caffe.io.resize_image(image, crop_dim)
        cluster_array[ind, :] = image
    face_mean = np.sum(face_array, axis=0)/len(face_jpgs_list)
    cluster_mean = np.sum(cluster_array, axis=0)/len(cluster_jpgs_list)
    face_mean = face_mean.transpose(2, 0, 1)
    cluster_mean = cluster_mean.transpose(2, 0, 1)
    np.save(os.path.join(out_path, 'face_mean.npy'), face_mean)
    np.save(os.path.join(out_path, 'cluster_mean.npy'), cluster_mean)


def fill_face_data(net, transformer, data_path):
    with open(data_path) as data_fd:
        lines_list = data_fd.readlines()
    face_jpgs_list = [line.strip().split()[0] for line in lines_list if line.strip().split()[1] is '1']
    net.blobs['data'].reshape(len(face_jpgs_list), 3, 227, 227)
    for ind, jpg in enumerate(face_jpgs_list):
        image = caffe.io.load_image(jpg)
        net.blobs['data'].data[ind, :] = transformer.preprocess('data', image)



def fill_cluster_data(net, transformer, data_path):
    with open(data_path) as data_fd:
        lines_list = data_fd.readlines()
    cluster_jpgs_list = [line.strip().split()[0] for line in lines_list if line.strip().split()[1] is '0']
    net.blobs['data'].reshape(len(cluster_jpgs_list), 3, 227, 227)
    for ind, jpg in enumerate(cluster_jpgs_list):
        image = caffe.io.load_image(jpg)
        net.blobs['data'].data[ind, :] = transformer.preprocess('data', image)


if __name__ == "__main__":
    train_txt_file = 'experiment_0/train.txt'
    if not os.path.isfile('experiment_0/face_mean.npy'):
        calculate_mean('experiment_0/train.txt', 'experiment_0/')

    caffe.set_mode_cpu()
    net = caffe.Net('experiment_0/deploy.prototxt', caffe.TRAIN)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('experiment_0/face_mean.npy'))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)



    print 'helld'
    pass

