import numpy as np
# a = np.array([[1, 2, 2], [2, 0, 0], [3, 4, 2]])
# b = np.argwhere(a[0] == 2)
# print(b.flatten())
# a = np.arange(3)
# c = np.arange(3, 6)
# b = np.zeros(10)
# b[a] = 1
# b[c] = 2
# print(b)
import copy
# a = [3, 4]
# x = [1, 2, a]
# y = copy.copy(x)
# x[2][0] = 9
# print(x, y)
from import_other_pythonpath.Student import *
s = Student('zhuzhenhan', 'sx')
import networkx as nx
import json
import matplotlib.pyplot as plt

# g = nx.complete_graph(4)
def READ_json_file(filepath):
    '''

    :param filename:
    :return:
    '''
    with open(filepath, 'rb') as fp:
        json_data = json.load(fp)
        print(type(json_data))
        node = []
        edge = []
        color_map = []
        for single_data in (json_data):
            node1 = (single_data['from']['id'])
            if node1 not in node:
                node.append(node1)
                if single_data['from']['label'] == '1':
                    color_map.append('red')
                elif single_data['from']['label'] == '2':
                    color_map.append('blue')
                elif single_data['from']['label'] == '3':
                    color_map.append('black')
                else:
                    color_map.append('yellow')
            node2 = single_data['to']['id']
            if node2 not in node:
                node.append(node2)
                if single_data['to']['label'] == '1':
                    color_map.append('red')
                elif single_data['to']['label'] == '2':
                    color_map.append('blue')
                elif single_data['to']['label'] == '3':
                    color_map.append('black')
                else:
                    color_map.append('yellow')
            edge.append(f'{node1}, {node2}, 1')
        print(len(node), len(color_map), len(edge))
        G = nx.DiGraph()
        G = nx.read_edgelist(edge, nodetype=None, data=[('weight', int)])
        nx.draw_networkx(G, with_labels=False, node_color=color_map)
        plt.show()
        return G
def get_pos_from_graph(G):
    pos = nx.nx_pydot.pydot_layout(G)
    print(pos)
def try_enumerate():
    for index, content in enumerate(np.arange(10)):
        print(index, content)
import tensorflow as tf
def try_tf_comcat():
    a = np.random.rand(3, 4, 2)
    b = np.random.rand(3, 4, 5)
    c = tf.concat([a, b[:, :, :-1]], 2)
    d = b[:, :, -1]
    print(c.shape, d.shape)
from tensorflow.keras.models import load_model
def model_summary():
    model = load_model('generator_training\\0830_wgan1st\\mse_generator.h5')
    model.summary()
    model = load_model('generator_training\\0830_wgan1st\\mse_discriminator.h5')
    model.summary()
import pickle
from layout_evaluate import topo_evaluate_batch, reproduce_graph_from_pos_adj, D1, D2
def train_graph_evaluate():
    file = open('train_data\\3000_graph.pkl', 'rb')
    all_graph = pickle.load(file)
    file.close()
    d1 = np.array([])
    d2 = np.array([])
    for graph in all_graph:
        d1 = np.append(d1, D1(graph))
        d2 = np.append(d2, D2(graph))
    plt.figure()
    plt.plot(d1, color='r', linestyle='-', label='d1')

    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()

    plt.figure()
    plt.plot(d1, color='b', linestyle='--', label='d2')
    plt.legend()
    plt.show()
def try_np_concatenate():
    a = np.array([])
    b = np.random.rand(4)
    c = np.concatenate([a , b], axis=0)
    print(c)
    d = np.random.rand(2)
    print(np.concatenate([c, d], axis=0))
    print(c, d)
def try_zip():
    a = np.random.rand(3, 4)
    b = np.random.rand(3, 2)
    for x, y in zip(a, b):
        print(x, y)
import networkx as nx
def try_plt_draw():
    g = nx.complete_graph(4)
    plt.figure()
    nx.draw_networkx(g)
    plt.savefig('de.jpg')
    g2 = nx.complete_graph(3)
    plt.figure()
    nx.draw_networkx(g2)
    plt.savefig('de2.jpg')
def np_append():
    a = [np.array([]) ] * 4
    a[0] = np.append(a[0], 1)
    print(a)
import os
def fun_pkl():
    ad1 = pickle.load(open('generator_training\\mse_gan\\0905_d1_d2\\ad_score1.pkl', 'rb'))
    ad2 = pickle.load(open('generator_training\\mse_gan\\0905_d1_d2\\ad_score2.pkl', 'rb'))
    for i in range(4):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure()
        plt.plot(ad1[i], color='r', label='D1')
        plt.plot(ad2[i], color='b', linestyle='-', label='D2')
        plt.title('分数变化')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.legend()
        plt.savefig(os.path.join('generator_training\\mse_gan\\0905_d1_d2', f'adj_score_{i}.png'))
from layout_evaluate import D1, D2, reproduce_graph_from_pos_adj
def user_study():
    pos = np.load(open(f'train_topo_pics\\100\\100_graph_pos.npy', 'rb'))
    adj = np.load(open(f'train_topo_pics\\100\\100_graph_adj.npy', 'rb'))
    s1 = D1(reproduce_graph_from_pos_adj(pos[0], adj[0]))
    s2 = D2(reproduce_graph_from_pos_adj(pos[0], adj[0]))
    s = (s1 + s2) / 2.
    print(s1, s2, s)
def onehot_to_xy_batch(batch_gen_pos):
    # 32 * 30 * 81
    all_pos = []
    for single_grahh in batch_gen_pos:
        # 30 * 81
        pos = onehot_to_xy(single_grahh)
        # print('single_graph:', single_grahh.shape)
        all_pos.append(tuple(pos))
    all_pos = np.array(all_pos)
    # if all_pos.shape != (32, 30, 2):
    #     print(all_pos.shape)
    return all_pos
def onehot_to_xy_batch(batch_pos, adj):
    output_pos = []

    batch_pos = np.array(batch_pos)

    for pos, ad in zip(batch_pos, adj):
        single_pos = onehot_to_xy(pos, ad)
        output_pos.append(single_pos)
    return output_pos
AD = [np.array([])] * 4
AD_SCORE2 = [np.array([])] * 4
AD_SCORE3 = [np.array([])] * 4
FULL_SCORE = [np.array([])] * 4
ROOT_INDEX = [0, 8, 9, 17, 18, 26, 27, 35, 36, 44, 45, 53, 54, 62, 63, 71, 72, 80]
ROOT_SWITCH_INDEX = []
MIDDLE_INDEX = []
CONTACT_INDEX = []
for i in range(9):
    ROOT_SWITCH_INDEX.append(i * 9 + 1)
    ROOT_SWITCH_INDEX.append(i * 9 + 7)
    MIDDLE_INDEX.append(i * 9 + 2)
    MIDDLE_INDEX.append(i * 9 + 6)
    CONTACT_INDEX.append(i * 9 + 3)
    CONTACT_INDEX.append(i * 9 + 4)
    CONTACT_INDEX.append(i * 9 + 5)
OTHER_INDEX = list(set(np.arange(81)) - set(ROOT_INDEX))
def onehot_to_xy_batch_train_reduction(batch_pos, adj):
    output_pos = []

    batch_pos = np.array(batch_pos)

    for pos, ad in zip(batch_pos, adj):
        single_pos = onehot_to_xy(pos, ad)
        output_pos.append(single_pos)
    return output_pos
def onehot_to_xy_train_reduction(all_onehot, adj):
    all_output = tf.nn.softmax(all_onehot, axis=-1)
    all_output = np.array(all_output)
    chessboard = np.zeros((30, 81))

    pos = []
    for node, onehot in enumerate(all_output):
        if adj[node][-1] == 1:
            onehot[OTHER_INDEX] = 0
            index = np.argmax(onehot)
            # print(index)
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9

            pos.append((x, y))
        elif adj[node][-1] == 2:
            other_index = list(set(np.arange(81)) - set(ROOT_SWITCH_INDEX))
            onehot[other_index] = 0
            index = np.argmax(onehot)
            # print(index)
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9

            pos.append((x, y))
        elif adj[node][-1] == 3:
            other_index = list(set(np.arange(81)) - set(MIDDLE_INDEX))
            onehot[other_index] = 0
            index = np.argmax(onehot)
            y = 8 - int(index / 9)
            x = index - (8 - y) * 9

            pos.append((x, y))
        elif adj[node][-1] == 4:
            other_index = list(set(np.arange(81)) - set(CONTACT_INDEX))
            onehot[other_index] = 0
            index = np.argmax(onehot)
            y = 8 - int(index / 9)
            x = index - (8 - y) * 9

            pos.append((x, y))
        else:
            print('WRONG NODE LABEL!!!.....')
            onehot[ROOT_INDEX] = 0
            onehot[ROOT_SWITCH_INDEX] = 0
            index = np.argmax(onehot)
            chessboard[node][index] = 1
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9
            pos.append((x, y))
    pos = np.array(pos)
    return pos
def onehot_to_xy(all_onehot, adj):
    '''

    :param onehot: 30 * 81
    :return:pos 30 * 2
    '''
    all_output = tf.nn.softmax(all_onehot, axis=-1)
    all_output = np.array(all_output)
    chessboard = np.zeros((30, 81))

    pos = []
    for node, onehot in enumerate(all_output):
        if adj[node][-1] == 1:
            onehot[OTHER_INDEX] = 0
            index = np.argmax(onehot)
            # print(index)
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9
            while (x, y) in pos:
                onehot[index] = 0
                index = np.argmax(onehot)
                y = 8 - int((index) / 9)
                x = index - (8 - y) * 9
            pos.append((x, y))
        elif adj[node][-1] == 2:
            other_index = list(set(np.arange(81)) - set(ROOT_SWITCH_INDEX))
            onehot[other_index] = 0
            index = np.argmax(onehot)
            # print(index)
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9
            while (x, y) in pos:
                onehot[index] = 0
                index = np.argmax(onehot)
                y = 8 - int((index) / 9)
                x = index - (8 - y) * 9
            pos.append((x, y))
        elif adj[node][-1] == 3:
            other_index = list(set(np.arange(81)) - set(MIDDLE_INDEX))
            onehot[other_index] = 0
            index = np.argmax(onehot)
            y = 8 - int(index / 9)
            x = index - (8 - y) * 9
            while (x, y) in pos:
                onehot[index] = 0
                index = np.argmax(onehot)
                y = 8 - int((index) / 9)
                x = index - (8 - y) * 9
            pos.append((x, y))
        elif adj[node][-1] == 4:
            other_index = list(set(np.arange(81)) - set(CONTACT_INDEX))
            onehot[other_index] = 0
            index = np.argmax(onehot)
            y = 8 - int(index / 9)
            x = index - (8 - y) * 9
            while (x, y) in pos:
                onehot[index] = 0
                index = np.argmax(onehot)
                y = 8 - int((index) / 9)
                x = index - (8 - y) * 9
            pos.append((x, y))
        else:
            print('WRONG NODE LABEL!!!.....')
            onehot[ROOT_INDEX] = 0
            onehot[ROOT_SWITCH_INDEX] = 0
            index = np.argmax(onehot)
            chessboard[node][index] = 1
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9
            pos.append((x, y))
    pos = np.array(pos)
    return pos
def score():
    adj3000 = np.load('train_data\\3000_adj.npy')
    chess_onehot = np.load('train_data\\3000_chessboard.npy')
    posxy3000 = onehot_to_xy_batch(chess_onehot)
    score = []
    for adj, pos in zip(adj3000, posxy3000):
        score1 = D1(reproduce_graph_from_pos_adj(pos, adj))
        # score2 = D2(reproduce_graph_from_pos_adj(pos, adj))
        # s = (score2 + score1) / 2.
        s = score1
        score.append(s)
    plt.figure()
    plt.plot(score , 'b', label='all_traingraph_scores')
    plt.xlabel('b')
    plt.ylabel('real_fake_score_distance')
    plt.legend(loc='best')
    plt.savefig(os.path.join('chessboard', 'all_traingraph_d1_scores.png'))
    plt.show()
def onehot2catalog():
    chess_onehot = np.load('train_data\\3000_chessboard.npy')
    print(chess_onehot.shape)
    chessboard_cate = []
    for chessboard in chess_onehot:
        # 30 *81
        single_graph = []
        for onehot in chessboard:
            index = np.argwhere(onehot==1).flatten()
            # print('index:', index.shape)
            if index not in np.arange(0, 81):
                print('index wrong:', index)
            if index in single_graph:
                print('index position repeat')
            single_graph.append(index)
        chessboard_cate.append(single_graph)
    catelog = np.array(chessboard_cate)
    print(catelog.shape)
    np.save('train_data\\3000_cheboard_catelog.npy', chessboard_cate)
def chessboard2disccrete():
    chessboard = np.load('train_data\\3000_chessboard.npy')
    discrete_pos = onehot_to_xy_batch(chessboard)
    print(discrete_pos.shape)
    print(np.max(discrete_pos), np.min(discrete_pos))
    np.save('train_data\\3000_discrete_pos.npy', discrete_pos)
from My_Graph_Class import My_Graph
from chess_GAN import draw_topo_from_pos
# def draw_topo_from_pos(pos, adj):
#     '''
#
#     :param pos:
#     :param ad:
#     :return:
#     '''
#     nodes_label = adj[:, -1]
#     ad = adj[:, :-1]
#     graph = nx.from_numpy_matrix(ad)
#
#     root = np.argwhere(nodes_label == 1).flatten()
#     root_switch = np.argwhere(nodes_label == 2).flatten()
#     middle = np.argwhere(nodes_label == 3).flatten()
#     contact_switch = np.argwhere(nodes_label == 4).flatten()
#
#     dict_pos = dict.fromkeys(np.arange(30))
#     for i in range(len(pos)):
#         dict_pos[i] = tuple(pos[i])
#     my_graph = My_Graph(np.arange(30), root, root_switch, middle, contact_switch, graph.edges, dict_pos)
#     my_graph.show_graph()
#
#     draw_topo_from_graph(my_graph)
def check_onehot2xy():
    all_chessboard = np.load('train_data\\3000_chessboard.npy')
    adj = np.load('train_data\\3000_adj.npy')
    pos = onehot_to_xy_batch(all_chessboard)
    for pos, adj in zip(pos, adj):
        draw_topo_from_pos(pos, adj)
        plt.show()
def test0916():
    gen = tf.keras.models.load_model('chessboard\\wgan\\wgan_30_2_discrete_pos_plt_6_lossxy_test\\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    train_pos = np.load('train_data\\3000_pos.npy')
    gen_pos = gen(adj)
    for i, adj in enumerate(adj[:50]):
        plt.figure()
        draw_topo_from_pos(gen_pos[i], adj)
        plt.savefig(f'test0916_mse\\{i}mse_generator_image.jpg')
        plt.show()
        plt.figure()
        draw_topo_from_pos(train_pos[i], adj)
        plt.savefig(f'test0916_mse\\{i}mse_train_image.jpg')
        plt.show()
    plt.show()
def test0916_pretrain():
    gen = tf.keras.models.load_model('chessboard\\wgan\\wgan_30_2_discrete_pos_plt_8_pretrain_model\\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    train_pos = np.load('train_data\\3000_pos.npy')
    gen_pos = gen(adj)
    for i, adj in enumerate(adj[:50]):
        plt.figure()
        draw_topo_from_pos(gen_pos[i], adj)
        plt.savefig(f'test0916_pretrain\\{i}mse_generator_image.jpg')
        plt.show()
        plt.figure()
        draw_topo_from_pos(train_pos[i], adj)
        plt.savefig(f'test0916_pretrain\\{i}mse_train_image.jpg')
        plt.show()
    plt.show()
def test0916_original_shuffle_wgan():
    gen = tf.keras.models.load_model('generator_training\\0904_shuffle_wgan\\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    train_pos = np.load('train_data\\3000_pos.npy')
    gen_pos = gen(adj)
    for i, adj in enumerate(adj[:50]):
        plt.figure()
        draw_topo_from_pos(gen_pos[i], adj)
        plt.savefig(f'test0916_original_shuffle_wgan\\{i}mse_generator_image.jpg')
        plt.show()
        plt.figure()
        draw_topo_from_pos(train_pos[i], adj)
        plt.savefig(f'test0916_original_shuffle_wgan\\{i}mse_train_image.jpg')
        plt.show()
    plt.show()
def test_mlp():
    mlp = tf.keras.models.load_model('chessboard\\mlp\\0907_mlp\\drop_out_l2_mlp.h5')
    adj = np.load('train_data\\3000_7-11_adj.npy')
    real_pos = np.load('train_data\\3000_7-11_pos.npy')
    gen_pos = mlp(adj[:20])
    print(gen_pos.shape)
    for i, adj in enumerate(adj[:20]):
        plt.figure()
        print(gen_pos[i])
        draw_topo_from_pos(gen_pos[i], adj)
        # plt.savefig(f'chessboard\\mlp\\0907_mlp\\{i}mlp_image.jpg')
        plt.show()
def smooth_curve_test():
    val = np.random.rand(100)
    factor = 0.8
    smooth = []
    for v in val:
        if smooth:
            pre = smooth[-1]
            smooth.append(pre * factor + v * (1 - factor))
        else:
            smooth.append(v)

    plt.figure()
    plt.plot(val, label='-', color='g')
    plt.plot(smooth, label='--', color='r')
    plt.show()
def multi_result():
    gen_model = tf.keras.models.load_model(r'chessboard\30_81\0925_all_D_1\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    real_test_adj = adj[0].reshape((1, 30, 31))
    label = real_test_adj[:, :, -1].reshape((1, 30, 1))
    for i in range(20):
        test_adj = real_test_adj[:, :, :-1] + np.random.random((1, 30, 30)) / 5.
        test_adj = tf.concat([test_adj, label], axis=-1)
        pos = gen_model(test_adj)
        print(pos.shape)
        pos = onehot_to_xy_batch(pos, real_test_adj)
        for p, a in zip(pos, real_test_adj):
            draw_topo_from_pos(p, a)
        plt.show()


if __name__ == '__main__':
    # g = READ_json_file('json\\1.json')
    # get_pos_from_graph(g)
    # try_enumerate()
    # model_summary()
    a = np.array([])
    # try_np_concatenate()
    # a = np.array([1, -3, 5, 6]).astype('float')
    # b = tf.nn.softmax(a, axis=-1)
    # print(b)
    # index = np.argmax(b)
    # print(b[index])
    # score()
    # a =np.random.rand(3, 2)
    # b = np.random.rand(3, 2)
    # MSE = tf.keras.losses.MeanSquaredError()
    # loss = MSE(a[:, :1], b[:, :1])
    # print(loss)
    # score()
    # a = np.random.rand(3, 2)
    # print(a)
    # a = tf.expand_dims(a, -1)
    # print(a.shape)
    # b = np.random.rand(3, 2, 3)
    # c = tf.concat([b, a], axis=-1)
    # print(c)
    # chessboard2disccrete()
    # onehot2catalog()
    # cate = np.load('train_data\\3000_cheboard_catelog.npy')
    #
    #
    # all_chess_81 = []
    # for graph_cate in cate:
    #     chess_81_discrete = np.zeros((81))
    #     print(graph_cate)
    #     for sort_index in graph_cate:
    #         chess_81_discrete[sort_index] = 1
    #     all_chess_81.append(chess_81_discrete)
    #     print(chess_81_discrete)
    # np.save('train_data\\3000_chess_81_discrete.npy

    # smooth_curve_test()
    # for i in range(5):
    #     a = np.zeros((1, 3, 4))
    #     a = a + np.random.random((1, 3, 4)) / 10.
    #     print(a)
    for i in range(-4, 4, 1):
        print(i)