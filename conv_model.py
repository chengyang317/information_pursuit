import conv_net
import dataset


class ConvModel(object):
    def __init__(self, images_percent, network_percent, batch_size):
        # self.work_path = '/scratch/dataset/information_pursue'
        # self.dataset_path = '/scratch/dataset/information_pursue'
        # self.images_path = '/scratch/dataset/256_ObjectCategories'
        self.work_path = '/scratch/yang/dataset/information_pursue'
        self.dataset_path = '/scratch/yang/dataset/information_pursue'
        self.images_path = '/scratch/yang/dataset/256_ObjectCategories'
        self.images_percent = images_percent
        self.network_percent = network_percent
        self.batch_size = batch_size
        self.image_shape = (227, 227, 3)
        self.image_classes = 256
        self.data_set = dataset.Dataset(images_path=self.images_path, work_path=self.work_path,
                                       batch_size=self.batch_size, images_percent=self.images_percent,
                                       image_shape= self.image_shape, image_classes=self.image_classes)
        self.conv_net = conv_net.ConvNet(batch_size=self.batch_size, network_percent=self.network_percent,
                                         work_path=self.work_path, data_set=self.data_set)

    def work(self):
        self.data_set.work()
        self.conv_net.work()

if __name__ == '__main__':
    conv_model = ConvModel(images_percent=0.05, network_percent=0.5, batch_size=30)
    conv_model.work()







