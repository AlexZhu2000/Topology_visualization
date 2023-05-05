import numpy as np
import os
NODES_50_MAX_X = 43
NODES_50_MAX_Y = 25
NODES_50_MAX_NODES_NUM = 60

NODES_100_MAX_X = 56
NODES_100_MAX_Y = 50
NODES_100_MAX_NODES_NUM = 110
'''
y 
|
|
|
|
|44                    87
|0                     43
-------------------------> x
'''
def fun_xy_to_index(x, y):
    index = x + y * NODES_50_MAX_X + 1
    return index

def xy_to_index(path,max_nodes_num, chessboard_length):

    # path = os.path.join('train_data', TP50)
    all_pos = np.load(path, allow_pickle=True)
    new_all_index_pos = []
    new_all_chessboard_pos = []
    for graph_pos in all_pos:
        new_graph_index_pos = []
        pad_num = 0
        for single_pos in graph_pos:
            index = fun_xy_to_index(single_pos[0], single_pos[1])
            new_graph_index_pos.append(index)
        if len(new_graph_index_pos) < max_nodes_num:
            pad_num = max_nodes_num - len(new_graph_index_pos)
        new_graph_index_pos = np.array(new_graph_index_pos)
        new_graph_index_pos = np.pad(new_graph_index_pos, (0, pad_num), 'constant', constant_values=-1)
        new_all_index_pos.append(new_graph_index_pos)
        '''
        根据index: (n ):[35, 334, 674, ..., 234]合成chessboard坐标 (n * 60)
        [[0, 0, 0, ..., 1, 0, 0, .., 0], [0, ..., 1,..., 0], ..., []]
        '''
        new_graph_chessboard_pos = []           # 单个图的chessborad型坐标
        for index in new_graph_index_pos:
            temp = np.zeros((chessboard_length, 1))
            if index >= 0:
                temp[index] = 1
            new_graph_chessboard_pos.append(temp)
        print(len(new_graph_chessboard_pos))
        new_all_chessboard_pos.append(new_graph_chessboard_pos)
    print('new_all_chessboard_pos shape:', np.array(new_all_chessboard_pos).shape)
    # np.save('train_data/tp50_nodes_719_index_pos.npy', np.array(new_all_index_pos))
    # np.save('train_data/tp')
    return np.array(new_all_index_pos), np.array(new_all_chessboard_pos)
def transfor_xy_to_chessboard(choose):
    TP50 = 'train_data/tp50_nodes_818_graph_normalize_pos.npy'
    TP100 = 'train_data/tp100_nodes_719_graph_normalize_pos.npy'
    if choose == 50:
        chessboard_length = (NODES_50_MAX_X + 1) * (NODES_50_MAX_Y + 1)
        max_nodes_num = NODES_50_MAX_NODES_NUM
        new_all_index_pos, new_all_chessboard_pos = xy_to_index(TP50, max_nodes_num, chessboard_length)
        np.save('train_data/tp50_chessboard_pos.npy', new_all_chessboard_pos)
    else:
        chessboard_length = (NODES_100_MAX_X + 1) * (NODES_100_MAX_Y + 1)
        max_nodes_num = NODES_100_MAX_NODES_NUM
        new_all_index_pos, new_all_chessboard_pos = xy_to_index(TP100, max_nodes_num, chessboard_length)
        np.save('train_data/tp100_chessboard_pos.npy', new_all_chessboard_pos)
def adj_padding(path, node_max_size):
    '''
    将adj补全
    :param path:
    :param node_size:
    :return:
    '''
    all_adj = np.load(path, allow_pickle=True)
    new_all_adj = []
    for single_adj in all_adj:
        label = single_adj[:, -1]
        single_ad = single_adj[:, :-1]
        node_num = len(single_adj)
        padding_num = node_max_size - node_num
        label = np.pad(label, ((0, padding_num), (0, 0)), 'constant', constant_values=-1)

        label = label.reshape(-1, 1)
        single_ad = np.pad(single_ad, ((0, padding_num), (0, padding_num)), 'constant', constant_values=(0, 0))
        new_single_adj = np.concatenate((single_ad, label), axis=1)
        new_all_adj.append(new_single_adj)
        print(new_single_adj.shape)
    print(np.array(new_all_adj).shape)
    return np.array(new_all_adj)
def transfor_adj_to_same(choose):
    TP50 = 'train_data/tp50_nodes_818_adj.npy'
    TP100 = 'train_data/tp100_nodes_719_adj.npy'
    if choose == 50:
        nodes_num = NODES_50_MAX_NODES_NUM
        new_all_adj = adj_padding(TP50, nodes_num)
        np.save('train_data/tp50_nodes_818_adj_padding.npy', new_all_adj)
    else:
        nodes_num = NODES_100_MAX_NODES_NUM
        new_all_adj = adj_padding(TP100, nodes_num)
        np.save('train_data/tp100_nodes_719_adj_padding.npy', new_all_adj)

if __name__ == '__main__':
    transfor_xy_to_chessboard(100)
    # a = np.array([1, 2, 3, 4 ,5])
    # a = np.pad(a, (0, 5), 'constant')
    # print(a)
    # a = np.ones((5, 5))
    # b = a[:, -1]
    #
    #
    # b = np.pad(b, ((0, 2)), 'constant')
    # print(b.shape)
    # b = b.reshape(-1, 1)
    # print(b, '\n', b.shape)
    # a = np.pad(a, ((0, 2), (0, 2)), 'constant', constant_values=0)
    #
    # c = np.concatenate((a, b), axis=1)
    # print(b, '\n', a)
    # print(c)
