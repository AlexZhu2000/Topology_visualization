import sys
import os
import tensorflow as tf
import numpy as np
TOPO_PATH = os.getcwd()

JSON_PATH = TOPO_PATH + '\\json\\'              # 该json文件的数据类型为List

if __name__ == '__main__':
    print('this is the main of PATH.py')
    a = np.arange(40).reshape(20, 2)
    b = np.arange(20)
    train_datasets = tf.data.Dataset.from_tensor_slices((a, b))
    train_datasets = train_datasets.shuffle(20).batch(4)
    for x, y in train_datasets.as_numpy_iterator():
        print(x, y)
    print(train_datasets)
    # train_datasets = train_datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)