def define_ConvNet_layers(shape, net_percent):
    layers_attrs = {'input_layer': {'layer_name': 'input', 'shape': shape},
                    'conv1_layer': {'layer_name': 'conv1', 'shape': [11, 11, 3, int(96 * net_percent)], 'stddev': 1e-2,
                                    'strides': [1, 4, 4, 1], 'padding': 'SAME', 'biase': 0.1},
                    'norm1_layer': {'layer_name': 'norm1', 'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75},
                    'pool1_layer': {'layer_name': 'pool1', 'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1],
                                    'padding': 'SAME'},
                    'conv2_layer': {'layer_name': 'conv2', 'shape': [5, 5, int(96 * net_percent), int(256 * net_percent)],
                                    'stddev': 1e-2, 'strides': [1, 2, 2, 1], 'padding': 'SAME', 'biase': 0.1},
                    'norm2_layer': {'layer_name': 'norm2', 'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75},
                    'pool2_layer': {'layer_name': 'pool2', 'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1],
                                    'padding': 'SAME'},
                    'conv3_layer': {'layer_name': 'conv3', 'shape': [3, 3, int(256 * net_percent), int(256 * net_percent)],
                                    'stddev': 1e-2, 'strides': [1, 2, 2, 1], 'padding': 'SAME', 'biase': 0.1},
                    'norm3_layer': {'layer_name': 'norm3', 'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75},
                    'pool3_layer': {'layer_name': 'pool3', 'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1],
                                    'padding': 'SAME'},
                    'full4_layer': {'layer_name': 'full4', 'shape': [None, int(4096 * net_percent)], 'stddev': 1e-2,
                                    'biase': 0.1},
                    'softmax5_layer': {'layer_name': 'softmax5', 'shape': [int(4096 * net_percent, None)],
                                       'stddev': 1e-2, 'biase': 0.1}
                    }
    return layers_attrs