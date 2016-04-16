import conv_model


if __name__ == '__main__':
    conv_model = conv_model.ConvModel(images_percent=0.4, network_percent=0.5, batch_size=30)
    conv_model.work()