import lmdb
import numpy as np
import pickle


env = lmdb.open('mylmdb', map_size=100000000000)

with env.begin(write=True) as txn:

    for i in range(10):
        a = dict()
        a['label'] = 1
        a['data'] = np.ones((227, 227, 3), dtype=np.float32)
        str_a = pickle.dumps(a)
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), str_a)