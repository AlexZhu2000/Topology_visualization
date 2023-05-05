import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

def MLP_30_81():
    adj = np.load('train_data\\3000_adj.npy')
    chess_onehot = np.load('train_data\\3000_discrete_pos.npy')
    model = mlp_model()
    model.compile(optimizer=keras.optimizers.Adam(),
                  # loss='categorical_crossentropy',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(adj, chess_onehot, batch_size=32, epochs=100, validation_batch_size=0.01)
    model.summary()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'], label='-')
    # plt.plot(history.history['val_loss'], label='--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    model.save('chessboard\\mlp\\0907_mlp\\drop_out_l2_sparse_categorical_crossentropy_mlp.h5')
    AD = []
    test_pos = model.predict(adj[:4])
    # all_posxy = onehot_to_xy_batch(test_pos)
    for i in range(test_pos.shape[0]):
        # proxy = onehot_to_xy(test_pos[i])
        # print(proxy.shape)
        print(test_pos[i])
        draw_topo_from_pos(test_pos[i], adj[i])
        AD.append(D1(reproduce_graph_from_pos_adj(test_pos[i], adj[i])))
        plt.show()

    plt.figure()
    plt.plot(AD, label='d1_score')
    plt.legend()
    plt.savefig(f'chessboard\\mlp\\0907_mlp\\d1_test_score.jpg')
    plt.show(block=False)

def mlp_model():
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, use_bias=True, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(2048, use_bias=True, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(30 * 2, use_bias=True, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Reshape((30, 2)))
    return model
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
        if (x, y) in pos:
            if (x - 0.5, y) not in pos:
                pos.append((x - 0.5, y))
            elif (x, y + 0.5) not in pos:
                pos.append((x, y + 0.5))
            elif (x + 0.5, y + 0.5) not in pos:
                pos.append((x + 0.5, y + 0.5))
            elif (x - 0.5, y + 0.5) not in pos:
                pos.append((x - 0.5, y + 0.5))
            elif (x - 0.5, y - 0.5) not in pos:
                pos.append((x - 0.5, y - 0.5))
            elif (x + 0.5, y - 0.5) not in pos:
                pos.append((x + 0.5, y - 0.5))
            elif (x + 0.5, y ) not in pos:
                pos.append((x + 0.5, y ))
            else:
                pos.append((x, y))
        else:
            pos.append((x, y))
        pos.append((x, y))
    pos = np.array(pos)
    return pos
from chess_mse_gan import draw_topo_from_pos
from layout_evaluate import D1, reproduce_graph_from_pos_adj
def MLP():
    adj = np.load('train_data\\3000_adj.npy')
    chess_onehot = np.load('train_data\\3000_discrete_pos.npy')
    model = mlp_model()
    model.compile(optimizer=keras.optimizers.Adam(),
                  # loss='categorical_crossentropy',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(adj, chess_onehot, batch_size=32, epochs=100, validation_batch_size=0.01)
    model.summary()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'], label='-')
    # plt.plot(history.history['val_loss'], label='--')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    model.save('chessboard\\mlp\\0907_mlp\\drop_out_l2_sparse_categorical_crossentropy_mlp.h5')
    AD = []
    test_pos = model.predict(adj[:4])
    # all_posxy = onehot_to_xy_batch(test_pos)
    for i in range(test_pos.shape[0]):
        # proxy = onehot_to_xy(test_pos[i])
        # print(proxy.shape)
        print(test_pos[i])
        draw_topo_from_pos(test_pos[i], adj[i])
        AD.append(D1(reproduce_graph_from_pos_adj(test_pos[i], adj[i])))
        plt.show()

    plt.figure()
    plt.plot(AD, label='d1_score')
    plt.legend()
    plt.savefig(f'chessboard\\mlp\\0907_mlp\\d1_test_score.jpg')
    plt.show(block=False)

def MLP_18_19():
    train_matrix, train_pos, root_onehot, _ = GG.Load_graph('graph_data\\all_graph_3000.pkl')
    # BUFFER_SIZE = 3000
    # BATCH_SIZE = 50
    # datasets = tf.data.Dataset.from_tensor_slices(train_matrix)
    # datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_pos = train_pos.astype('float64')
    train_matrix = train_matrix.astype('float64')
    root_onehot = root_onehot.astype('float64')
    train_matrix = tf.concat([train_matrix, root_onehot], 2)

    model = mlp_model_18_19()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['accuracy'])
    model.fit(train_matrix, train_pos, batch_size=100, epochs=200, validation_split=0.01)
    model.summary()
    model.save('graph_data\\Layout_Generator_2.h5')
    ad_matrix, _, root_onehot, all_color_map = GG.Load_graph('graph_data\\all_graph_3.pkl')
    ad_matrix = ad_matrix.astype('float64')
    root_onehot = root_onehot.astype('float64')
    test_matrix = tf.concat([ad_matrix, root_onehot], 2)
    # 保持和MSE一致便于比较
    pos = np.load('graph_data\\18-19pos.npy')
    ad_matrix = np.load('graph_data\\18-19adjacency.npy')
    test_pos = model.predict(ad_matrix[:4])
    test_matrix = ad_matrix[:4]
    # test_pos = model.predict(test_matrix)
    print(test_pos.shape)
    # # test_pos = test_pos.numpy()
    # test_matrix = test_matrix.numpy()
    for i in range(test_pos.shape[0]):
        fig = plt.figure(figsize=(10, 10))
        MSE_GAN.draw_topo_from_pos(test_pos[i], test_matrix[i])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'0813_MLP_res\\{i}-200.jpg')
        # g1 = nx.from_numpy_matrix(ad_matrix[i])
        # dict1 = dict.fromkeys(np.arange(18))
        # for v in range(18):
        #     dict1[v] = tuple(test_pos[i][v])
        #
        # nx.draw_networkx(g1, pos=dict1, node_color=all_color_map[i])
        plt.show()

if __name__ == '__main__':
    MLP()