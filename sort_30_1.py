
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pickle
import os
import networkx as nx
import numpy as np
from My_Graph_Class import My_Graph
import nodes_connections_layout as Nodes_Connection

def load_train_data():

    adj = np.load('train_data\\3000_adj.npy')
    chess_cate = np.load('train_data\\3000_cheboard_catelog.npy')
    return chess_cate.astype('float32'), adj.astype('float32')
PLT_SAVE_FILE = 'sort_30_1\\0916_30_1_2'
global POS, ADJACENCY
POS, ADJACENCY = load_train_data()
print(POS.shape)
MAX = np.max(POS)
POS = POS / MAX
BUFFER_SIZE = 3000
BATCH_SIZE = 32
train_datasets = tf.data.Dataset.from_tensor_slices((POS, ADJACENCY))
# train_datasets = train_datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generator_model():
    model = keras.Sequential()
    # (n, 30 * 31)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512, use_bias=False))  # (n, 400)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(256,  use_bias=False))      # (n, 400)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(128, use_bias=False))                           # (n, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(30, use_bias=False, activation='tanh'))    # (n, 30)
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((30, 1)))                                     # (n, 18, 2)

    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    # model.add(tf.keras.layers.Dense(32, use_bias=False))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(16, use_bias=False))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Softmax())

    model.add(tf.keras.layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


DIS_LOSS = np.array([])
GEN_LOSS = np.array([])
SCORE_LOSS = np.array([])
MLP_LOSS = np.array([])
def discriminator_loss(real_out, real_pos, real_ad, fake_out, fake_pos, fake_ad):
    # gan
    # real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    # fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    global DIS_LOSS
    #wgan的尝试
    real_loss = -tf.reduce_mean(real_out)
    fake_loss = tf.reduce_mean(fake_out)

    # mse-gan
    # real_loss = MSE(topo_evaluate_batch(real_pos, real_ad), real_out)
    # fake_loss = MSE(topo_evaluate_batch(fake_pos, fake_ad), fake_out)
    # fake_loss = MSE(tf.zeros_like(fake_out), fake_out)
    # print('\ndiscriminator_loss:', real_loss + fake_loss, '...')
    DIS_LOSS = np.append(DIS_LOSS, real_loss + fake_loss)
    return real_loss + fake_loss
MSE = tf.keras.losses.MeanSquaredError()

def generator_loss(fake_out, pos, ad, real_pos):
    global GEN_LOSS, SCORE_LOSS, MLP_LOSS

    # gan
    # loss1 = cross_entropy(tf.ones_like(fake_out), fake_out)

    # wgan
    loss1 = -tf.reduce_mean(fake_out)
    loss2 = MSE(real_pos, pos)
    # mse-gan
    # loss1 = MSE(tf.ones_like(s), fake_out)
    # print('\t \t \t \tgenerator_loss:', loss1 ,'\t','Evaluate loss:',loss2,  '...\n')
    GEN_LOSS = np.append(GEN_LOSS, loss1)
    MLP_LOSS = np.append(MLP_LOSS, loss2)
    # SCORE_LOSS = np.append(SCORE_LOSS, loss2)
    return loss1 + loss2

discriminator_opt = tf.keras.optimizers.Adam(1e-4)
# discriminator_opt = tf.keras.optimizers.RMSprop(0.01)
generator_opt = tf.keras.optimizers.Adam(1e-4)
# generator_opt = tf.keras.optimizers.RMSprop(0.01)
generator = generator_model()
discriminator = discriminator_model()

def train_step(pos, ad):
    '''
    每个batch有32组数据
    :param pos:32 * 30 * 81
    :param ad: 32 * 30 * 31
    :return:
    '''
    # noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # print(pos.shape)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # real_condition_pair = tf.concat([pos, ad[:, :, :-1]], 2)
        # real_out = discriminator(real_condition_pair, training=True)
        #
        # gen_position = generator(ad, training=True)
        # fake_condition_pair = tf.concat([gen_position, ad[:, :, :-1]], 2)
        # fake_out = discriminator(fake_condition_pair, training=True)
        # pos = tf.expand_dims(pos, -1)
        # real_pos = tf.concat([ad, pos],axis=-1)
        # real_out = discriminator(real_pos, training=True)            #real_pos (30 * 32)
        #
        # gen_position = generator(ad, training=True)                 # gen_position (30)
        # fake_pos = tf.concat([ad, tf.expand_dims(gen_position, -1)], axis=-1)
        # fake_out = discriminator(fake_pos, training=True)
        real_out = discriminator(pos, training=True)
        gen_pos = generator(ad, training=True)
        fake_pos = gen_pos
        fake_out = discriminator(fake_pos, training=True)
        fake_ad = ad
        real_pos = pos
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
    Nodes_Connection.draw_topo_from_graph(my_graph)
from layout_evaluate import D1, D2,reproduce_graph_from_pos_adj
AD = [np.array([])] * 4
AD_SCORE2 = [np.array([])] * 4
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
        pos.append((x, y))
    pos = np.array(pos)
    return pos
def cate_to_xy_v1(cate):
    '''
    30 * 1
    :param cate:(30, 1)    0 ~ 1
    :return:
    '''
    cate = cate * 80
    pos = np.zeros((30, 2))
    # print(cate)
    arr = [(position, index) for index, position in enumerate(cate)]
    arr.sort()
    # print(arr)
    for sort_index, (values, node_index) in enumerate(arr):
        # print(sort_index)
        y = 8 - int((sort_index) / 9)
        # print(y)
        x = sort_index - (8 - y) * 9
        if x not in np.arange(0, 9) or y not in np.arange(0, 9):
            print('wrong...')
        pos[node_index] = [x, y]
    return pos
def cate_to_xy(cate):
    '''

    :param cate: (30 * 1)  -1 ~ 1
    :return:
    '''
    cate = (cate + 1) * 40
    pos_index = np.around(cate)
    pos_xy = []
    for sort_index in pos_index:
        y = 8 - int(sort_index / 9)
        x = sort_index - (8 - y) * 9
        if (x, y) in pos_xy:
            if (x - 0.5, y) not in pos_xy:
                pos_xy.append((x - 0.5, y))
            elif (x, y + 0.5) not in pos_xy:
                pos_xy.append((x, y + 0.5))
            elif (x + 0.5, y + 0.5) not in pos_xy:
                pos_xy.append((x + 0.5, y + 0.5))
            elif (x - 0.5, y + 0.5) not in pos_xy:
                pos_xy.append((x - 0.5, y + 0.5))
            elif (x - 0.5, y - 0.5) not in pos_xy:
                pos_xy.append((x - 0.5, y - 0.5))
            elif (x + 0.5, y - 0.5) not in pos_xy:
                pos_xy.append((x + 0.5, y - 0.5))
            elif (x + 0.5, y ) not in pos_xy:
                pos_xy.append((x + 0.5, y ))
            else:
                pos_xy.append((x, y))
        else:
            pos_xy.append((x, y))
        pos_xy.append((x, y))
    return np.array(pos_xy)
def generate_plot_position(gen_model, generate_ad, n):
    # pre_images = gen_model(test_noise, trainable=False)
    generate_pos = gen_model(generate_ad)
    print(generate_pos)
    # batch_size * 30
    # print(f'-------------\n{generate_pos}\n--------------\n')
    fig = plt.figure(figsize=(10, 10))
    # all_graph = []
    for i in range(generate_pos.shape[0]):
        plt.subplot(2, 2, i + 1)
        # plt.imshow((generate_pos[i, :, :] + 1) / 2, cmap='gray')
        # 这里是因为我们使用tanh激活函数之后会将结果限制在-1到1之间，而我们需要将其转化到0到1之间。
        proxy = cate_to_xy(generate_pos[i])
        draw_topo_from_pos(proxy, generate_ad[i])
        #AD[i] = np.append(AD[i], D1(reproduce_graph_from_pos_adj(proxy, generate_ad[i])))
        # AD_SCORE2[i] = np.append(AD_SCORE2[i], D2(reproduce_graph_from_pos_adj(proxy, generate_ad[i])))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'{PLT_SAVE_FILE}\\my_gan{n}.jpg')
    plt.show(block=False)
    # plt.pause(1)Z
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
EPOCH = 200

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

    plt.figure()
    plt.plot(MLP_LOSS, 'b', label='MLP_loss')
    plt.xlabel('b')
    plt.ylabel('MLP_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'MLP.png'))

    plt.show()

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

