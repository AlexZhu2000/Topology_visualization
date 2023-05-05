import tensorflow as tf
import numpy as np



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
def onehot_to_xy(all_onehot):
    '''

    :param onehot: 30 * 81
    :return:pos 30 * 2
    '''
    all_output = tf.nn.softmax(all_onehot, axis=-1)
    chessboard = np.zeros((30, 81))
    pos = []
    for node, onehot in enumerate(all_output):
        index = np.argmax(onehot)
        chessboard[node][index] = 1
        y = 8 - int((index) / 9)
        x = index - (8 - y) * 9
        # if (x, y) in pos:
        #     if (x - 0.5, y) not in pos:
        #         pos.append((x - 0.5, y))
        #     elif (x, y + 0.5) not in pos:
        #         pos.append((x, y + 0.5))
        #     elif (x + 0.5, y + 0.5) not in pos:
        #         pos.append((x + 0.5, y + 0.5))
        #     elif (x - 0.5, y + 0.5) not in pos:
        #         pos.append((x - 0.5, y + 0.5))
        #     elif (x - 0.5, y - 0.5) not in pos:
        #         pos.append((x - 0.5, y - 0.5))
        #     elif (x + 0.5, y - 0.5) not in pos:
        #         pos.append((x + 0.5, y - 0.5))
        #     elif (x + 0.5, y ) not in pos:
        #         pos.append((x + 0.5, y ))
        #     else:
        #         pos.append((x, y))
        # else:
        #     pos.append((x, y))
        pos.append((x, y))
    pos = np.array(pos)
    return pos

from layout_evaluate import D1, D2, reproduce_graph_from_pos_adj
import matplotlib.pyplot as plt
generator = tf.keras.models.load_model('chessboard\\mse_gan\\0906_chess_mse_gan_2\\mse_generator.h5')
adj3000 = np.load('train_data\\3000_adj.npy')
chess_onehot = np.load('train_data\\3000_chessboard.npy')
gen_pos = onehot_to_xy_batch(generator(adj3000))

posxy3000 = onehot_to_xy_batch(chess_onehot)

import os

def get_gen_graph_score():
    gen_socre = []
    for adj, pos in zip(adj3000, gen_pos):
        print(pos.shape)
        score1 = D1(reproduce_graph_from_pos_adj(pos, adj))
        score2 = D2(reproduce_graph_from_pos_adj(pos, adj))
        s = (score2 + score1) / 2.
        gen_socre.append(s)

    gen_socre = np.array(gen_socre)
    distance = train_score - gen_socre
    plt.figure()
    plt.plot(distance, 'b', label='train_gen_dis')
    plt.xlabel('b')
    plt.ylabel('gen_loss')
    plt.legend(loc='best')
    plt.savefig(('train_gen_dis.png'))

def get_all_train_score():
    train_score = []
    for adj, pos in zip(adj3000, posxy3000):
        # print(pos.shape)
        score1 = D1(reproduce_graph_from_pos_adj(pos, adj))
        # score2 = D2(reproduce_graph_from_pos_adj(pos, adj))
        # s = (score2 + score1) / 2.
        s = score1
        train_score.append(s)
    train_score = np.array(train_score)
    plt.figure()
    plt.plot(train_score, 'b', label='train_new_d1_score')
    plt.xlabel('b')
    plt.ylabel('gen_loss')
    plt.legend(loc='best')
    plt.savefig(('all_train_new_d1_score.png'))
    plt.show()
def get_all_gen_pos_score():
    gen_pos = onehot_to_xy_batch(generator(adj3000))
    gen_score = []
    for adj, pos in zip(adj3000, gen_pos):
        score1 = D1(reproduce_graph_from_pos_adj(pos, adj))
        s = score1
        gen_score.append(s)
    plt.figure()
    plt.plot(gen_score, 'b', label='gen_pos_d1_score')
    plt.xlabel('b')
    plt.ylabel('gen_score')
    plt.legend(loc='best')
    plt.savefig(('gen_pos_d1_score.png'))
    plt.show()
if __name__ == '__main__':
    # get_all_train_score()
    get_all_gen_pos_score()