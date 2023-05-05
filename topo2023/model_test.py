import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from constrain_pos import *
CHESSBOARD_SIZE = NODES_50_X_NUM * NODES_50_Y_NUM
import networkx as nx
from My_graph_class5 import *
def draw_topo_from_pos(pos, adj):
    '''

    :param pos:
    :param ad:
    :return:
    '''
    adj = np.array(adj)
    nodes_label = adj[:, -1]
    ad = adj[:, :-1]
    graph = nx.from_numpy_matrix(ad)
    # print(adj, nodes_label)
    root = np.argwhere(nodes_label == 1).flatten()
    root_switch = np.argwhere(nodes_label == 3).flatten()
    middle = np.argwhere(nodes_label == 4).flatten()
    contact_switch = np.argwhere(nodes_label == 2).flatten()
    busbar = np.argwhere(nodes_label == 6).flatten()
    dict_pos = dict.fromkeys(np.arange(adj.shape[0]))
    # print(root, root_switch, busbar)
    for i in range(len(pos)):
        dict_pos[i] = tuple(pos[i])
    my_graph = My_Graph(np.arange(adj.shape[0]), root, busbar, root_switch, middle, contact_switch, graph.edges, dict_pos)
    my_graph.show_graph()
    plt.show()
    # Nodes_Connection.draw_topo_from_graph(my_graph)
def get_mask_batch(adj_batch):
    '''
    带padding的adj
    :param adj_batch:
    :return:
    '''
    mask_batch = []
    for adj in adj_batch:
        mask = get_mask(adj)
        mask_batch.append(mask)
    return np.array(mask_batch)
def get_mask(adj):
    mask_batch = []

    single_mask = []

    for node, adj_label in enumerate(adj):
        if adj_label[-1] == 1:
            # 如果为根节点，对应Mask设置为1
            temp = np.zeros(CHESSBOARD_SIZE)
            temp[ROOT_INDEX] = 1
            single_mask.append(temp)
        elif adj_label[-1] == 2:
            temp = np.zeros(CHESSBOARD_SIZE)
            temp[CONTACT_INDEX] = 1
            single_mask.append(temp)
        elif adj_label[-1] == 3:
            temp = np.zeros(CHESSBOARD_SIZE)
            temp[OUT_SWITCH_INDEX] = 1
            single_mask.append(temp)
        elif adj_label[-1] == 4:
            temp = np.zeros(CHESSBOARD_SIZE)
            temp[MIDDLE_INDEX] = 1
            single_mask.append(temp)
        elif adj_label[-1] == 6:
            temp = np.zeros(CHESSBOARD_SIZE)
            temp[BUSBAR_INDEX] = 1
            single_mask.append(temp)
        elif adj_label[-1] == -1:
            # print('WRONG NODE LABEL IN TRAIN CONSTRAIN....')
            temp = np.zeros(CHESSBOARD_SIZE)
            single_mask.append(temp)
    # mask_batch.append(single_mask)
    # mask_batch = np.array(mask_batch).astype('float32')
    # mask_batch = tf.constant(mask_batch)

    return np.array(single_mask)
np.set_printoptions(threshold=sys.maxsize)
def train_score():
    train_pos = np.load('train_data/tp50_chessboard_pos.npy')

    for single in train_pos:

        all_output = tf.nn.softmax(single)

        for node in all_output:
            print(np.argwhere(node == 1))
            index = np.argmax(node)
import json
def output_single_json(single_adj, single_pos, score):
    nodes = []
    edges = []

    all_label = single_adj[:, -1]
    # for index, node in enumerate(single_adj):
    for index in np.arange(len(all_label)):
        nodes.append({'id': str(index), 'pos': list(single_pos[index].astype('float')), 'label': str(all_label[index])})
        print(single_adj.shape, index, all_label[index])
        g = nx.from_numpy_matrix(single_adj[:, :-1])
        for edge in g.edges():
            edges.append({'from': edge[0], 'to': edge[1]})
    dict = {'nodes': nodes, 'edges': edges}
    with open(f'for_tomax\\0_{score}.json', "w") as f:
        json.dump(dict, f)
def test_generator():
    generator = keras.Sequential()
    # (n, 60 * 61)
    generator.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(4096,  use_bias=False))      # (n, 400)
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    # model.add(tf.keras.layers.Dense(16384, use_bias=False))                           # (n, 256)
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    # model.add(tf.keras.layers.Dense(32768, use_bias=False))  # (n, 256)
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    generator.add(tf.keras.layers.Dense(68640, use_bias=False))  # (n, 30)
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.LeakyReLU())
    generator.add(tf.keras.layers.Reshape((60, 1144)))
    generator.build(input_shape=(None, 60, 61))
    generator.load_weights('epoch30mse_generator.h5')
    ad = np.load('train_data\\tp50_nodes_818_adj_padding.npy')
    train_pos = np.load('train_data/tp50_chessboard_pos.npy')
    test_ad_padding = ad[:4]
    train_pos4 = train_pos[:4]
    train_pos4 = np.squeeze(train_pos4, axis=-1)
    test_pos_padding = generator(test_ad_padding)
    mask = get_mask_batch(test_ad_padding)
    # print('mask:', mask.shape)
    constrain_pos = tf.multiply(test_pos_padding, mask)
    print(constrain_pos.shape, test_pos_padding.shape, train_pos4.shape)
    #print('test_pos:', constrain_pos.shape)
    pos_xy = np.array(constrain_onehot_to_xy_batch(constrain_pos, test_ad_padding))
    train_pos4_xy = np.array(constrain_onehot_to_xy_batch(train_pos4, test_ad_padding))

    #print('pos_xy', pos_xy.shape)
    new_adj = []
    new_pos = []
    index = 0
    for (single_adj_padding, single_pos) in zip(test_ad_padding, pos_xy):
        new_single_adj = []

        all_label = single_adj_padding[:, -1]
        single_ad_padding = single_adj_padding[:, :-1]
        padding_num = 0
        for node in all_label:
            if node == -1:
                padding_num += 1
        new_single_pos = single_pos[:-padding_num, :]
        new_single_adj = np.concatenate((single_ad_padding[:-padding_num, :-padding_num], all_label[:-padding_num].reshape(-1, 1)), axis=-1)
        test_score = topo_evaluate_single(new_single_pos, new_single_adj)
        print('test_score', test_score)
        output_single_json(new_single_adj, new_single_pos, test_score)
        # train_single_pos = train_pos4_xy[index][:-padding_num, :]
        # train_score = topo_evaluate_single(train_single_pos, new_single_adj)
        # print('train_score', train_score)

        # !后加的
        # test_el = cal_edge_length(new_single_pos)
        # train_el = cal_edge_length(train_single_pos)
        # test_mean_edge_length = mean_el(test_el, new_single_adj)
        # train_mean_edge_length = mean_el(train_el, new_single_adj)
        # test_distri = distribution(new_single_adj, test_el)
        # train_distri = distribution(new_single_adj, train_el)
        # print(f'test, 平均线段长度: {test_mean_edge_length}, 变电站周围节点分布方差: {test_distri}')
        # print(f'train, 平均线段长度: {train_mean_edge_length}, 变电站周围节点分布方差: {train_distri}')

        # print('new_single_adj', new_single_adj.shape)
        # print(new_single_pos)
        # draw_topo_from_pos(new_single_pos, new_single_adj)
        # draw_topo_from_pos(train_single_pos, new_single_adj)
        new_adj.append(new_single_adj)
        new_pos.append(new_single_pos)
        index += 1
    # new_adj= np.array(new_adj)
    # new_pos = np.array(new_pos)
    # print(new_adj.shape, new_pos.shape)

    # for pos, ad in zip(new_pos, new_adj):
    #     draw_topo_from_pos(pos, ad)
    #     plt.show()
from layout_evaluate import topo_evaluate_batch, topo_evaluate_single
# def try1():
#
#     s = topo_evaluate_single(pos_xy, ad)


def cal_edge_length(pos):
    """
    计算所有节点互相相连的长度
    :param pos:
    :return:
    """
    return [[np.linalg.norm(pos[i] - pos[j]) for i in range(len(pos))] for j in range(len(pos))]


def mean_el(el, adj):
    """
    计算所有边的平均长度
    :param el:
    :param adj:
    :return:
    """
    edge_cnt = 0
    edge_length = 0
    for i in range(len(el)):
        for j in range(len(el[i])):
            if adj[i][j] == 0:
                continue
            edge_cnt += 1
            edge_length += el[i][j]
    return edge_length / edge_cnt


def distribution(adj, el):
    """
    以变电站为中心，每个节点加入最相近的变电站中心并且计数，求节点数量分布的方差（感觉好像没啥用 ）
    :param adj:
    :param el:
    :return:
    """
    root_set = []
    for i, node in enumerate(adj):
        if node[-1] == 1:
            root_set.append(i)

    count = [0 for i in range(len(root_set))]

    for i in range(len(adj)):
        min_index = -1
        for index, root in enumerate(root_set):
            if min_index == -1 or el[index][i] < el[min_index][i]:
                min_index = index
        count[min_index] += 1
    print('distribution:', count)
    return np.var(np.array(count))
def for_tomax0324():
    generator = keras.Sequential()
    # (n, 60 * 61)
    generator.add(tf.keras.layers.Flatten())
    generator.add(tf.keras.layers.Dense(68640, use_bias=False))  # (n, 30)
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.LeakyReLU())
    generator.add(tf.keras.layers.Reshape((60, 1144)))
    generator.build(input_shape=(None, 60, 61))
    generator.load_weights('epoch75mse_generator.h5')

    ad = np.load('train_data/tp50_nodes_818_adj_padding.npy')
    train_pos = np.load('train_data/tp50_chessboard_pos.npy')
    test_ad_padding = ad[:1]
    train_pos4 = train_pos[:1]
    train_pos4 = np.squeeze(train_pos4, axis=-1)
    test_pos_padding = generator(test_ad_padding)
    mask = get_mask_batch(test_ad_padding)
    # print('mask:', mask.shape)
    constrain_pos = tf.multiply(test_pos_padding, mask)
    print(constrain_pos.shape, test_pos_padding.shape, train_pos4.shape)
    # print('test_pos:', constrain_pos.shape)
    pos_xy = np.array(constrain_onehot_to_xy_batch(constrain_pos, test_ad_padding))
    train_pos4_xy = np.array(constrain_onehot_to_xy_batch(train_pos4, test_ad_padding))

    # print('pos_xy', pos_xy.shape)
    new_adj = []
    new_pos = []
    for (single_adj_padding, single_pos) in zip(test_ad_padding, pos_xy):
        new_single_adj = []

        all_label = single_adj_padding[:, -1]
        single_ad_padding = single_adj_padding[:, :-1]
        padding_num = 0
        for node in all_label:
            if node == -1:
                padding_num += 1
        new_single_pos = single_pos[:-padding_num, :]
        new_single_adj = np.concatenate(
            (single_ad_padding[:-padding_num, :-padding_num], all_label[:-padding_num].reshape(-1, 1)), axis=-1)
        test_score = topo_evaluate_single(new_single_pos, new_single_adj)
        print('test_score', test_score)
        output_single_json(new_single_adj, new_single_pos, test_score)

if __name__ == '__main__':
    for_tomax0324()
    # a = np.zeros((3, 5))
    # a[0][0] = 1
    # print(a)
    # all_output = tf.nn.softmax(a)
    # print(all_output)