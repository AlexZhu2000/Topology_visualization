
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pickle
import os
import networkx as nx
import numpy as np
from My_Graph_Class import My_Graph
import nodes_connections_layout as Nodes_Connection
from layout_evaluate import topo_evaluate_batch, D1, D2, reproduce_graph_from_pos_adj

def load_train_data():

    adj = np.load('train_data\\3000_adj.npy')
    chess_onehot = np.load('train_data\\3000_chessboard.npy')
    return chess_onehot, adj
PLT_SAVE_FILE = 'chessboard\\mse_gan\\0907_d1_only_root'
global POS, ADJACENCY
POS, ADJACENCY = load_train_data()
MAX = np.max(POS)
POS = POS / MAX
BUFFER_SIZE = 3000
BATCH_SIZE = 32
train_datasets = tf.data.Dataset.from_tensor_slices((POS, ADJACENCY))
# train_datasets_batch = train_datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generator_model():
    model = keras.Sequential()
    # (n, 30 * 81)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024,  use_bias=False))      # (n, )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(2048, use_bias=False))                           # (n, )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(30 * 81 , use_bias=False, activation='tanh'))    # (n, )
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Reshape((30, 81)))                                     # (n,)

    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(256, use_bias=False, activation='tanh'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


DIS_LOSS = np.array([])
GEN_LOSS = np.array([])
SCORE_LOSS = np.array([])
REAL_FAKE_SCORE_DIFF = np.array([])
MLP_LOSS = np.array([])
def discriminator_loss(real_out, real_pos, real_ad, fake_out, fake_pos, fake_ad):
    # gan
    # real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    # fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    global DIS_LOSS, REAL_FAKE_SCORE_DIFF
    #wgan的尝试
    # real_loss = -tf.reduce_mean(real_out)
    # fake_loss = tf.reduce_mean(fake_out)
    # mse-gan
    score_real = topo_evaluate_batch(real_pos, real_ad)
    score_fake = topo_evaluate_batch(fake_pos, fake_ad)
    real_loss = MSE(score_real, real_out)
    fake_loss = MSE(score_fake, fake_out)
    diff = score_real - score_fake
    print(diff.shape, np.mean(diff))
    REAL_FAKE_SCORE_DIFF = np.append(REAL_FAKE_SCORE_DIFF, np.mean(diff))

    # fake_loss = MSE(tf.zeros_like(fake_out), fake_out)
    # print('\ndiscriminator_loss:', real_loss + fake_loss, '...')
    DIS_LOSS = np.append(DIS_LOSS, real_loss + fake_loss)
    return real_loss + fake_loss
MSE = tf.keras.losses.MeanSquaredError()

def generator_loss(fake_out, pos, ad, real_pos ):
    global GEN_LOSS, SCORE_LOSS, MLP_LOSS

    # gan
    # loss1 = cross_entropy(tf.ones_like(fake_out), fake_out)

    # wgan
    # loss1 = -tf.reduce_mean(fake_out)

    # mse-gan

    # print(s)
    s = topo_evaluate_batch(pos, ad)
    loss1 = MSE(tf.ones_like(s), fake_out)
    loss2 = MSE(real_pos[:, :1].astype('float32'), pos[:, :1].astype('float32'))
    # print('\t \t \t \tgenerator_loss:', loss1 ,'\t','Evaluate loss:',loss2,  '...\n')
    GEN_LOSS = np.append(GEN_LOSS, loss1)
    SCORE_LOSS = np.concatenate([SCORE_LOSS, s], axis=0)
    MLP_LOSS = np.append(MLP_LOSS, loss2)
    return 0.3 *loss1 + 0.7 * loss2

discriminator_opt = tf.keras.optimizers.Adam(1e-4)
generator_opt = tf.keras.optimizers.Adam(1e-4)
generator = generator_model()
discriminator = discriminator_model()

def train_step(pos, ad):
    '''
    每个batch有32组数据
    :param pos:
    :param ad:32 *  30 * 31
    :return:
    '''

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator(pos, training=True)
        gen_position = generator(ad, training=True)
        fake_out = discriminator(gen_position, training=True)

        # gen_position (batch_size , 30, 81)
        fake_pos = onehot_to_xy_batch(gen_position)
        # fake_pos (batch_size, 30, 2)
        fake_ad = ad
        real_pos = onehot_to_xy_batch(pos)

        real_ad = ad
        gen_loss = generator_loss(fake_out, fake_pos, fake_ad, real_pos)
        disc_loss = discriminator_loss(real_out, real_pos, real_ad, fake_out, fake_pos, fake_ad)
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))

generate_ad = ADJACENCY[:4]

def draw_topo_from_pos(pos, adj):
    '''

    :param pos:
    :param ad:
    :return:
    '''
    nodes_label = adj[:, -1]
    ad = adj[:, :-1]
    graph = nx.from_numpy_matrix(ad)

    root = np.argwhere(nodes_label == 1).flatten()
    root_switch = np.argwhere(nodes_label == 2).flatten()
    middle = np.argwhere(nodes_label == 3).flatten()
    contact_switch = np.argwhere(nodes_label == 4).flatten()

    dict_pos = dict.fromkeys(np.arange(30))
    for i in range(len(pos)):
        dict_pos[i] = tuple(pos[i])
    my_graph = My_Graph(np.arange(30), root, root_switch, middle, contact_switch, graph.edges, dict_pos)
    # my_graph.show_graph()
    Nodes_Connection.draw_topo_from_graph(my_graph)
AD = [np.array([])] * 4
AD_SCORE2 = [np.array([])] * 4
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
def generate_plot_position(gen_model, generate_ad, n):
    # pre_images = gen_model(test_noise, trainable=False)
    generate_pos = gen_model(generate_ad)
    # print(f'-------------\n{generate_pos}\n--------------\n')
    fig = plt.figure(figsize=(10, 10))
    # all_graph = []
    for i in range(generate_pos.shape[0]):
        plt.subplot(2, 2, i + 1)
        proxy = onehot_to_xy(generate_pos[i])
        # print(proxy.shape)
        draw_topo_from_pos(proxy, generate_ad[i])
        AD[i] = np.append(AD[i], D1(reproduce_graph_from_pos_adj(proxy, generate_ad[i])))
        AD_SCORE2[i] = np.append(AD_SCORE2[i], D2(reproduce_graph_from_pos_adj(proxy, generate_ad[i])))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'{PLT_SAVE_FILE}\\my_gan{n}.jpg')
    plt.show(block=False)
    # plt.close()
    # plt.pause(1)

EPOCH = 100

def draw_train_loss():
    plt.figure()
    plt.plot(DIS_LOSS, 'b', label = 'dis_loss')
    plt.xlabel('b')
    plt.ylabel('dis_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'dis_loss.png'))

    plt.figure()
    plt.plot(GEN_LOSS, 'b', label='gen_loss')
    plt.xlabel('b')
    plt.ylabel('gen_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'gen_loss.png'))

    plt.figure()
    plt.plot(SCORE_LOSS, 'b', label='score_loss')
    plt.xlabel('b')
    plt.ylabel('score_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'score_loss.png'))

    np.save('real_fake_score_distance.npy', REAL_FAKE_SCORE_DIFF)
    plt.figure()
    plt.plot(REAL_FAKE_SCORE_DIFF, 'b', label='real_fake_score_distance')
    # plt.xlabel('b')
    # plt.ylabel('real_fake_score_distance')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'real_fake_score_distance.png'))

    MLP_LOSS
    plt.figure()
    plt.plot(MLP_LOSS, 'b', label='MLP_loss')
    # plt.xlabel('b')
    # plt.ylabel('real_fake_score_distance')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'mlp.png'))
    plt.show()

def draw_test_ad_score():
    for i in range(4):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure()
        plt.plot(AD[i], color='r', label='D1')
        plt.plot(AD_SCORE2[i], color='b', linestyle='-', label='D2')
        plt.title('分数变化')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.legend()
        plt.savefig(os.path.join(PLT_SAVE_FILE, f'adj_score_{i}.png'))
    # for i in range(4):
    #     plt.rcParams['font.sans-serif'] = ['SimHei']
    #     plt.rcParams['axes.unicode_minus'] = False
    #     plt.figure()
    #     plt.plot(AD_SCORE2[i], color='r', label='D2')
    #     plt.title('d2分数变化')
    #     plt.xlabel('轮次')
    #     plt.ylabel('分数')
    #     plt.legend()
    #     plt.savefig(os.path.join(PLT_SAVE_FILE, f'adj_score2_{i}.png'))
    pickle.dump(AD, open(f'{PLT_SAVE_FILE}\\ad_score1.pkl', 'wb'))
    pickle.dump(AD_SCORE2, open(f'{PLT_SAVE_FILE}\\ad_score2.pkl', 'wb'))
def train(train_datasets, epochs):
    for i in range(epochs):
        train_datasets_batch = train_datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        for pos_batch, ad_batch in train_datasets_batch:
            train_step(pos_batch, ad_batch)

        print(f'EPOCH: {i}...\n')
        generate_plot_position(generator, generate_ad, i)

if __name__ == '__main__':
    train(train_datasets, EPOCH)
    generator.save(f'{PLT_SAVE_FILE}\\mse_generator.h5')
    discriminator.save(f'{PLT_SAVE_FILE}\\mse_discriminator.h5')
    draw_train_loss()
    draw_test_ad_score()

