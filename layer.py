

def define_ConvNet_layers(shape, net_percent):
    layers_attrs = {'input_layer': {'layer_name': 'input', 'shape': shape},
                    'conv1_layer': {'layer_name': 'conv1', 'shape': [11, 11, 3, int(96 * net_percent)], 'stddev': 1e-2,
                                    'strides': [1, 4, 4, 1], 'padding': 'SAME', 'biase': 0.1},
                    'norm2_layer': {'layer_name': 'norm2', 'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75},
                    'pool3_layer': {'layer_name': 'pool3', 'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1],
                                    'padding': 'SAME'},
                    'conv4_layer': {'layer_name': 'conv4', 'shape': [5, 5, int(96 * net_percent), int(256 * net_percent)],
                                    'stddev': 1e-2, 'strides': [1, 2, 2, 1], 'padding': 'SAME', 'biase': 0.1},
                    'norm5_layer': {'layer_name': 'norm5', 'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75},
                    'pool6_layer': {'layer_name': 'pool6', 'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1],
                                    'padding': 'SAME'},
                    'conv7_layer': {'layer_name': 'conv7', 'shape': [3, 3, int(256 * net_percent), int(256 * net_percent)],
                                    'stddev': 1e-2, 'strides': [1, 2, 2, 1], 'padding': 'SAME', 'biase': 0.1},
                    'norm8_layer': {'layer_name': 'norm8', 'depth_radius': 5, 'bias': 1.0, 'alpha': 1e-4, 'beta': 0.75},
                    'pool9_layer': {'layer_name': 'pool9', 'ksize': [1, 3, 3, 1], 'strides': [1, 2, 2, 1],
                                    'padding': 'SAME'},
                    }