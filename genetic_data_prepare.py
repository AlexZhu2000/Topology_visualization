import numpy as np
import tensorflow as tf

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
def onehot_to_xy_batch(batch_pos, adj):
    output_pos = []

    batch_pos = np.array(batch_pos)

    for pos, ad in zip(batch_pos, adj):
        single_pos = onehot_to_xy(pos, ad)
        output_pos.append(single_pos)
    return output_pos
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
def data_100_for_genetic(number):
    gen_model = tf.keras.models.load_model(r'chessboard\30_81\0925_all_D_1\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    train_pos = np.load('train_data\\3000_chessboard.npy')

    all_adj_genetic_pos = []
    for adj_i in range(20):
        print(adj_i)
        real_test_adj = adj[adj_i].reshape((1, 30, 31))
        label = real_test_adj[:, :, -1].reshape((1, 30, 1))
        genetic_pos = []
        for j in range(number):
            data_choice = np.arange(1, 5).astype('float32')
            beta = np.random.choice(data_choice, 1)
            test_adj = real_test_adj[:, :, :-1] + np.random.random((1, 30, 30)) / beta
            test_adj = tf.concat([test_adj, label], axis=-1)
            pos = gen_model(test_adj)
            pos = onehot_to_xy_batch(pos, real_test_adj)
            print(pos.shape)
            genetic_pos.append(pos)
        all_adj_genetic_pos.append(genetic_pos)
    np.save('train_data\\3000_100_genetic_pos.npy', np.array(all_adj_genetic_pos))
    print(np.array(all_adj_genetic_pos).shape)


if __name__ == '__main__':
    # data_100_for_genetic(100)
    adj = np.load('train_data\\3000_adj.npy')
    a = adj[:20]
    print(a.shape)
    #np.save('train_data\\20_30x31_adj.npy', a)