

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from layout_evaluate import *
def evaluate_model():
    model = keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, use_bias=True, activation='tanh'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, use_bias=True, activation='tanh'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(32, use_bias=True, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    return model
def train_30_81_D3_evaluate_model():
    gen_model_1 = tf.keras.models.load_model(r'chessboard\30_81\0922_30_81_full_constrain_1\mse_generator.h5')
    gen_model_2 = tf.keras.models.load_model(r'chessboard\30_81\0921_30_81_train_mse_constrain_2\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    good_pos = np.load('train_data\\3000_chessboard.npy')
    bad_pos_1 = gen_model_1(adj)
    bad_pos_2 = gen_model_2(adj)

def train_evaluate_model():
    gen_model_1 = tf.keras.models.load_model('generator_training\\mse_gan\\0905_d1_d2\\mse_generator.h5')
    gen_model_2 = tf.keras.models.load_model(r'chessboard\wgan\wgan_30_2_discrete_pos_plt_8_pretrain_model\mse_generator.h5')
    adj = np.load('train_data\\3000_adj.npy')
    adj = tf.convert_to_tensor(adj)
    good_pos = np.load('train_data\\3000_pos.npy')
    bad_pos_1 = gen_model_1(adj)
    bad_pos_2 = gen_model_2(adj)
    bad_pos = tf.concat([bad_pos_1, bad_pos_2], axis=0)
    bad_adj = tf.concat([adj, adj], axis=0)
    # 6000 * 30 * 2
    good_train = tf.concat([adj, good_pos], axis=-1)
    good_score = topo_evaluate_batch(good_pos, adj)
    bad_train = tf.concat([bad_adj, bad_pos], axis=-1)
    bad_score = topo_evaluate_batch(bad_pos, bad_adj)

    print(good_train.shape, good_score.shape, bad_train.shape, bad_score.shape)

    train_x = tf.concat([good_train, bad_train], axis=0)
    train_y = tf.concat([good_score, bad_score], axis=0)
    print(train_x.shape, train_y.shape)

    # BUFFER_SIZE = 3000
    # BATCH_SIZE = 32
    # train_datasets = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    model = evaluate_model()
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=32, epochs=50, validation_split=0.01)
    model.summary()
    model.save(r'topo_evaluate_model\score3_30_2_model.h5')

    test_pos = np.load('train_data\\3000_discrete_pos.npy')
    right_score = topo_evaluate_batch(test_pos, adj)
    right_score.reshape((3000, 1))
    test_x = tf.concat([adj, test_pos], axis=-1)
    predict_score = model.predict(test_x)

    distance = right_score - predict_score
    plt.figure()
    plt.plot(distance, label='right - predict')
    plt.legend(loc='best')
    plt.show()
def test_evaluate_model():
    model = tf.keras.models.load_model('topo_evaluate_model\\score3_30_2_model.h5')
    test_pos = np.load('train_data\\3000_discrete_pos.npy')
    adj = np.load('train_data\\3000_adj.npy')
    adj = tf.convert_to_tensor(adj)
    right_score = topo_evaluate_batch(test_pos, adj).reshape((3000, 1))
    print(right_score.shape)
    test_x = tf.concat([adj, test_pos], axis=-1)
    predict_score = model.predict(test_x)

    distance = right_score - predict_score
    print(distance.shape)
    plt.figure()
    plt.plot(distance, label='right - predict')
    plt.legend(loc='best')
    plt.show()
def find_experience_D3_Ascore():
    adj_3000 = np.load('train_data\\3000_adj.npy')
    pos_3000 = np.load('train_data\\3000_pos.npy')
    D3_score = []
    for pos, adj in zip(pos_3000, adj_3000):
        D3_score.append(D3(reproduce_graph_from_pos_adj(pos, adj)))
    print(np.mean(np.array(D3_score)))
    plt.figure()
    plt.plot(D3_score, color='r')
    plt.xticks([])
    plt.show()
    # 训练数据的所有D3评分普遍分布在 1.4--1.7，最大值2，最小值1.2

if __name__ == '__main__':
    test_evaluate_model()