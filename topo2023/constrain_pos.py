import numpy as np
import sys
import tensorflow as tf
np.set_printoptions(threshold=sys.maxsize)
from xy_to_onehot import NODES_50_MAX_X, NODES_50_MAX_Y, NODES_50_MAX_NODES_NUM
NODES_50_X_NUM = NODES_50_MAX_X + 1
NODES_50_Y_NUM = NODES_50_MAX_Y +1
ROOT_INDEX = []
CONTACT_INDEX = []
OUT_SWITCH_INDEX = []
MIDDLE_INDEX = []
BUSBAR_INDEX = []
for i in range(NODES_50_MAX_Y + 1):
    ROOT_INDEX.append(i * NODES_50_X_NUM + 0)
    ROOT_INDEX.append(i * NODES_50_X_NUM + 1)
    ROOT_INDEX.append(i * NODES_50_X_NUM + 43)
    ROOT_INDEX.append(i * NODES_50_X_NUM + 42)

    BUSBAR_INDEX.append(i * NODES_50_X_NUM + 2)
    BUSBAR_INDEX.append(i * NODES_50_X_NUM + 3)
    BUSBAR_INDEX.append(i * NODES_50_X_NUM + 41)
    BUSBAR_INDEX.append(i * NODES_50_X_NUM + 40)

    OUT_SWITCH_INDEX.append(i * NODES_50_X_NUM + 4)
    OUT_SWITCH_INDEX.append(i * NODES_50_X_NUM + 5)
    OUT_SWITCH_INDEX.append(i * NODES_50_X_NUM + 39)
    OUT_SWITCH_INDEX.append(i * NODES_50_X_NUM + 38)
    for j in range(6, 38):
        MIDDLE_INDEX.append(i * NODES_50_X_NUM + j)
CONTACT_INDEX = MIDDLE_INDEX
# a = np.zeros((NODES_50_MAX_X + 1) * (NODES_50_MAX_Y +1))
# a[ROOT_INDEX + CONTACT_INDEX + OUT_SWITCH_INDEX + MIDDLE_INDEX + BUSBAR_INDEX] = 1
# print(a)
def constrain_onehot_to_xy_batch(constrain_pos, adj):
    output = []
    constrain_pos = np.array(constrain_pos)
    for pos, ad in zip(constrain_pos, adj):
        single_pos = constrain_onehot_to_xy(pos)
        output.append(single_pos)
    return output
def constrain_onehot_to_xy(all_onehot):
    '''

    :return:
    '''
    all_output = tf.nn.softmax(all_onehot)
    all_output = np.array(all_output)

    pos = []
    for node, onehot in enumerate(all_output):
        index = np.argmax(onehot)
        y = int(index / NODES_50_X_NUM)
        x = index - y * NODES_50_X_NUM
        while (x, y) in pos:
            #如果位置已被使用，则顺序搜寻一次

            onehot[index] = 0
            index = np.argmax(onehot)
            y = int(index / NODES_50_X_NUM)
            x = index - y * NODES_50_X_NUM
        pos.append((x, y))
    return np.array(pos)
if __name__ == '__main__':
    index = np.array([2, 4])
    print(np.argmax(index))
    a = np.zeros((3, 2,1))
    a = np.squeeze(a, -1)
    print(a.shape)
    print(tf.__version__)
    adj = np.load('train_data\\tp50_nodes_818_adj_padding.npy', allow_pickle=True)
    chess_cate = np.load('train_data\\tp50_chessboard_pos.npy', allow_pickle=True)
    chess_cate = np.squeeze(chess_cate, -1)
    for ad in adj:
        for node in ad:
            print(node.shape)
            if node[-1] not in [1, 2, 3, 4, 6]:
                print(node[-1])

    # i2 = [0]
    # p = np.zeros(5)
    # p[index + i2] = 1
    # print(p)