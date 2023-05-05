import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def onehot_to_xy_batch(batch_pos, adj):
    output_pos = []

    batch_pos = np.array(batch_pos)

    for pos, ad in zip(batch_pos, adj):
        single_pos = onehot_to_xy_1(pos, ad)
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
def onehot_to_xy_1(all_onehot, adj):
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
            while (x, y) in pos:
                onehot[index] = 0
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
            while (x, y) in pos:
                onehot[index] = 0
                index = np.argmax(onehot)
                # print(index)
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
                # print(index)
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
from layout_evaluate import *
import os
def score():
    adj3000 = np.load('train_data\\3000_adj.npy')
    chess_onehot = np.load('train_data\\3000_chessboard.npy')
    posxy3000 = onehot_to_xy_batch(chess_onehot, adj3000)
    score = []
    for adj, pos in zip(adj3000, posxy3000):
        score1 = D1(reproduce_graph_from_pos_adj(pos, adj))
        score2 = D2(reproduce_graph_from_pos_adj(pos, adj))
        score3 = D3(reproduce_graph_from_pos_adj(pos, adj))
        s = (score2 + score1 + score3)
        score.append(s)
    plt.figure()
    plt.plot(score , label='all_traingraph_scores')
    plt.xlabel('b')
    plt.ylabel('real_fake_score_distance')
    plt.legend(loc='best')
    plt.savefig(os.path.join('chessboard', 'all_traingraph_scores.png'))
    plt.show()
from chess_GAN import draw_topo_from_pos
from layout_evaluate import topo_evaluate_batch
import time
PLT_SAVE_PATH = 'lianxian_pics\\D1+D2+D3\\'
def multi_result():
    gen_model = tf.keras.models.load_model(r'chessboard\30_81\0925_all_D_1\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    train_pos = np.load('train_data\\3000_chessboard.npy')
    print(train_pos.shape)
    for adj_i in range(10):
        real_test_adj = adj[adj_i].reshape((1, 30, 31))
        single_train_pos = train_pos[adj_i].reshape((1, 30, 81))
        single_train_pos = onehot_to_xy_batch(single_train_pos, real_test_adj)
        label = real_test_adj[:, :, -1].reshape((1, 30, 1))
        max_score = -1
        for i in range(10000):
            a = np.arange(1, 5).astype('float32')
            deta = np.random.choice(a, 1)
            test_adj = real_test_adj[:, :, :-1] + np.random.random((1, 30, 30)) / deta
            test_adj = tf.concat([test_adj, label], axis=-1)
            pos = gen_model(test_adj)
            pos = onehot_to_xy_batch(pos, real_test_adj)
            topo_adj = tf.convert_to_tensor(real_test_adj)
            score = topo_evaluate_batch(pos, topo_adj)
            # plt.figure()
            # draw_topo_from_pos(np.squeeze(pos), np.squeeze(real_test_adj))
            # plt.xlabel(str(score))
            # plt.show()
            if score > max_score:
                max_pos = pos

                max_score = score
            max_adj = real_test_adj

        for p, a in zip(max_pos, max_adj):
            # plt.figure(figsize=(20, 10))
            # plt.subplot(1, 2, 1)
            print(p.shape, a.shape)
            draw_topo_from_pos(p, a)
            # plt.xlabel(str(max_score))
            fig_id = np.random.rand()
            plt.savefig(PLT_SAVE_PATH + f'{adj_i}_gen_{max_score}.jpg')
            plt.show()
            train_score = topo_evaluate_batch(single_train_pos, topo_adj)
            # plt.xlabel((str(train_score)))
            draw_topo_from_pos(np.squeeze(single_train_pos), np.squeeze(real_test_adj))
            plt.savefig(PLT_SAVE_PATH + f'{adj_i}_train_{train_score}.jpg')
            plt.show()
        # plt.savefig(f'chessboard\\30_81\\multi_select_same_pos\\{adj_i}.jpg')

def onehot_to_xy_batch_train_reduction(batch_pos, adj):
    output_pos = []

    batch_pos = np.array(batch_pos)

    for pos, ad in zip(batch_pos, adj):
        single_pos = onehot_to_xy_train_reduction(pos, ad)
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
def reduction_train_pics():
    adj = np.load('train_data\\3000_adj.npy')
    train_chessboard = np.load('train_data\\3000_chessboard.npy')
    train_posxy = onehot_to_xy_batch_train_reduction(train_chessboard, adj)
    i = 0
    for p, a in zip(train_posxy, adj):
        draw_topo_from_pos(p, a)
        i += 1
        plt.savefig(f'train_topo_pics\\3000_chessboard_reduction\\{i}.jpg')
        plt.show()
if __name__ == '__main__':
    # multi_result()
    score()