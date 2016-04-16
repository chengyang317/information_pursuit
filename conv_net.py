import framwork
import tensorflow as tf
from net import Net
import layer


class ConvNet(Net):
    def __init__(self, net_name, batch_size, net_device, net_percent, work_path, data_set):
        super(ConvNet, self).__init__(net_name, batch_size, net_device, net_percent, work_path, data_set)
        self.layer_attrs = layer.define_ConvNet_layers(shape=(self.batch_size,) + self.image_shape,
                                                       net_percent=self.net_percent)
        self.train_loop = 10000

    def build_network(self):
        net_tensors = self.net_tensors
        layer_attrs = self.layer_attrs
        with self.net_graph.as_default(), tf.device(self.net_device):
            net_tensors.update(framwork.add_input_layer(layer_attrs['input_layer']))

            net_tensors.update(framwork.add_conv_layer(net_tensors['input_images'], layer_attrs['conv1_layer']))
            net_tensors.update(framwork.add_norm_layer(net_tensors['conv1_relu'], layer_attrs['norm1_layer']))
            net_tensors.update(framwork.add_pool_layer(net_tensors['norm1_norm'], layer_attrs['pool1_layer']))

            net_tensors.update(framwork.add_conv_layer(net_tensors['pool1_pool'], layer_attrs['conv2_layer']))
            net_tensors.update(framwork.add_norm_layer(net_tensors['conv2_relu'], layer_attrs['norm2_layer']))
            net_tensors.update(framwork.add_pool_layer(net_tensors['norm2_norm'], layer_attrs['pool2_layer']))

            net_tensors.update(framwork.add_conv_layer(net_tensors['pool2_pool'], layer_attrs['conv3_layer']))
            net_tensors.update(framwork.add_norm_layer(net_tensors['conv3_relu'], layer_attrs['norm3_layer']))
            net_tensors.update(framwork.add_pool_layer(net_tensors['norm3_norm'], layer_attrs['pool3_layer']))

            net_tensors.update(framwork.add_full_layer(net_tensors['pool3_pool'], layer_attrs['full4_layer']))

            if not layer_attrs['softmax5_layer']['shape'][1]:
                layer_attrs['softmax5_layer']['shape'][1] = self.data_set.image_classes
            net_tensors.update(framwork.add_softmax_layer(net_tensors['full4_relu'], layer_attrs['softmax5_layer']))

            inputs = [net_tensors['softmax5_op'], net_tensors['input_labels']]
            net_tensors.update(framwork.add_cross_entropy_layer(inputs, layer_attrs['loss_layer']))

            net_tensors.update(framwork.add_train_layer(layer_attrs['loss_op'], layer_attrs['train_layer']))

            inputs = [net_tensors['softmax5_op'], net_tensors['input_labels']]
            net_tensors.update(framwork.add_train_layer(inputs, layer_attrs['eval_layer']))

            net_tensors.update(framwork.add_train_layer(layer_attrs['aux_layer']))

    def train_network(self):
        with self.net_graph.as_default(), tf.device(self.net_device):
            sess = self.net_sess
            images = self.net_tensors['input_images']
            labels = self.net_tensors['input_labels']
            train_op = self.net_tensors['train_op']
            loss = self.net_tensors['loss_cross_entropy_mean']
            total_loss = self.net_tensors['loss_sum']
            loss1 = self.net_tensors['loss_conv1_kernel']
            loss2 = self.net_tensors['loss_conv2_kernel']
            loss3 = self.net_tensors['loss_conv3_kernel']
            loss4 = self.net_tensors['loss_full4_kernel']
            kernel4 = self.net_tensors['full4_kernel']
            loss5 = self.net_tensors['loss_softmax5_kernel']
            soft_max = self.net_tensors['softmax5_op']

            input_dict = {}
            for step in xrange(self.train_loop):
                image_datas, image_labels = self.fetch_batch_datas(self.data_set.train_que)
                input_dict.update({images: image_datas, labels: image_labels})
                # _, total_loss_value = sess.run([train_op, total_loss], feed_dict=input_dict)
                _, k4, l1, l2, l3, l4, l5, loss_value, total_loss_value = sess.run(
                    [train_op, kernel4, loss1, loss2, loss3, loss4, loss5, loss, total_loss], feed_dict=input_dict)
                if step % 20 == 0:
                    print('step is %d, total_loss is %f, loss is %f' % (step, total_loss_value, loss_value))
                if step % 500 == 0 and step != 0:
                    self.eval_network()
                if step % 1000 == 0 and step != 0:
                    self.save_network()



