
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
    chess_cate = np.load('train_data\\3000_discrete_pos.npy')
    return chess_cate.astype('float32'), adj.astype('float32')
PLT_SAVE_FILE = 'chessboard\\condition_gan\\0915_1'
test_adj = np.load('train_data\\3000_7-11_adj.npy')
test_adj = test_adj[:4]
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
    model.add(tf.keras.layers.Dense(512,  use_bias=False))      # (n, 400)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(256, use_bias=False))                           # (n, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(30 * 2, use_bias=False, activation='tanh'))    # (n, 30)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((30, 2)))                                     # (n, 18, 2)

    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())
    #
    # model.add(tf.keras.layers.Dense(8, use_bias=False))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Softmax())

    model.add(tf.keras.layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


DIS_LOSS = np.array([])
GEN_LOSS = np.array([])
SCORE_LOSS = np.array([])
MLP_LOSS = np.array([])
MSE_DIS_LOSS = np.array([])
MSE_GEN_LOSS =np.array([])
WGAN_DIS_LOSS = np.array([])
WGAN_GEN_LOSS = np.array([])

from layout_evaluate import topo_evaluate_batch
def discriminator_loss(real_out, fake_out):
    # gan
    # real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    # fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    real_loss = MSE(real_out, tf.ones_like(real_out))
    fake_loss = MSE(fake_out, tf.ones_like(fake_out))
    global DIS_LOSS, MSE_DIS_LOSS, WGAN_DIS_LOSS
    #wgan的尝试
    # real_loss = -tf.reduce_mean(real_out)
    # fake_loss = tf.reduce_mean(fake_out)
    # mse-gan
    # mse_real_loss = MSE(topo_evaluate_batch(real_pos, real_ad), 1.)
    # mse_fake_loss = MSE(topo_evaluate_batch(fake_pos, fake_ad), 0.)
    # fake_loss = MSE(topo_evaluate_batch(fake_pos, fake_ad), fake_out)
    # fake_loss = MSE(tf.zeros_like(fake_out), fake_out)
    # print('\ndiscriminator_loss:', real_loss + fake_loss, '...')
    # WGAN_DIS_LOSS = np.append(WGAN_DIS_LOSS, real_loss + fake_loss)
    # MSE_DIS_LOSS = np.append(MSE_DIS_LOSS, mse_real_loss +mse_fake_loss)
    DIS_LOSS = np.append(DIS_LOSS, real_loss + fake_loss )
    return real_loss + fake_loss
MSE = tf.keras.losses.MeanSquaredError()

def generator_loss(fake_out):
    global GEN_LOSS, SCORE_LOSS, MLP_LOSS, MSE_GEN_LOSS, WGAN_GEN_LOSS

    # gan
    # loss1 = cross_entropy(tf.ones_like(fake_out), fake_out)
    # mse_loss = cross_entropy(pos, real_pos)
    # wgan
    # loss1 = -tf.reduce_mean(fake_out)
    loss1 = MSE(fake_out, tf.ones_like(fake_out))
    # WGAN_GEN_LOSS = np.append(WGAN_GEN_LOSS, loss1)
    # lossx = MSE(real_pos[:, :, 0], pos[:, :, 0])
    # lossy = MSE(real_pos[:, :, 1], pos[:, :, 1])
    # mse-gan
    # s = topo_evaluate_batch(pos, ad)
    # mse_gen_loss = MSE(tf.ones_like(s), s)
    # print('\t \t \t \tgenerator_loss:', loss1 ,'\t','Evaluate loss:',loss2,  '...\n')
    GEN_LOSS = np.append(GEN_LOSS, loss1)
    # MSE_GEN_LOSS = np.append(MSE_GEN_LOSS, mse_gen_loss)
    # MLP_LOSS = np.append(MLP_LOSS, lossx + lossy)
    # SCORE_LOSS = np.append(SCORE_LOSS, loss2)
    return loss1

discriminator_opt = tf.keras.optimizers.Adam(1e-4)
# discriminator_opt = tf.keras.optimizers.RMSprop(0.0001)
generator_opt = tf.keras.optimizers.Adam(1e-4)
# generator_opt = tf.keras.optimizers.RMSprop(0.0001)
generator = generator_model()
discriminator = discriminator_model()

def train_step(pos, ad):
    '''
    每个batch有32组数据
    :param pos:32 * 30 * 2
    :param ad: 32 * 30 * 31
    :return:
    '''
    # noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # print(pos.shape)
    batch_size = pos.shape[0]
    noise = tf.random.normal([batch_size, 30, 1])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # real_pos + adj
        real_condition_pair = tf.concat([ad, pos], -1)
        real_out = discriminator(real_condition_pair, training=True)
        # fake_pos + adj
        gen_input = tf.concat([ad, noise], -1)
        gen_position = generator(gen_input, training=True)
        fake_condition_pair = tf.concat([ad, gen_position], -1)
        fake_out = discriminator(fake_condition_pair, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))

generate_ad = test_adj

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
def cate_to_xy(cate):
    '''

    :param cate:
    :return:
    '''
def generate_plot_position(gen_model, generate_ad, n):
    # pre_images = gen_model(test_noise, trainable=False)
    batch_size = generate_ad.shape[0]
    noise = tf.random.normal([batch_size, 30, 1])
    gen_input = tf.concat([generate_ad, noise], -1)
    generate_pos = gen_model(gen_input)
    print(f'-------------\n{generate_pos}\n--------------\n')
    fig = plt.figure(figsize=(10, 10))
    # all_graph = []
    for i in range(generate_pos.shape[0]):
        plt.subplot(2, 2, i + 1)
        # plt.imshow((generate_pos[i, :, :] + 1) / 2, cmap='gray')
        # 这里是因为我们使用tanh激活函数之后会将结果限制在-1到1之间，而我们需要将其转化到0到1之间。

        draw_topo_from_pos(generate_pos[i], generate_ad[i])
        AD[i] = np.append(AD[i], D1(reproduce_graph_from_pos_adj(generate_pos[i], generate_ad[i])))
        AD_SCORE2[i] = np.append(AD_SCORE2[i], D2(reproduce_graph_from_pos_adj(generate_pos[i], generate_ad[i])))
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
EPOCH = 100

def draw_train_loss():
    plt.figure()
    plt.plot(DIS_LOSS, 'b', label = 'dis_loss')
    plt.xlabel('b')
    plt.ylabel('dis_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'dis_loss.png'))

    plt.figure()
    plt.plot(MSE_GEN_LOSS, 'b', label='MSE_GEN')
    plt.xlabel('b')
    plt.ylabel('MSE_GEN')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'MSE_GEN.png'))

    plt.figure()
    plt.plot(MSE_DIS_LOSS, 'b', label='MSE_DIS')
    plt.xlabel('b')
    plt.ylabel('MSE_DIS')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'MSE_DIS.png'))

    plt.figure()
    plt.plot(WGAN_DIS_LOSS, 'b', label='WGAN_DIS')
    plt.xlabel('b')
    plt.ylabel('WGAN_DIS')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'WGAN_DIS.png'))

    plt.figure()
    plt.plot(WGAN_GEN_LOSS, 'b', label='WGAN_GEN')
    plt.xlabel('b')
    plt.ylabel('WGAN_GEN')
    plt.legend(loc='best')
    plt.savefig(os.path.join(PLT_SAVE_FILE, 'WGAN_GEN.png'))

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
def test_generator():
    generator = tf.keras.models.load_model('wgan_30_2_discrete_pos_plt_5_lossxy\\mse_generator.h5')
    ad = np.load('..\\..\\train_data\\3000_adj.npy')
    test_ad = ad[:12]
    test_pos = generator(test_ad)
    for pos, ad in zip(test_pos, test_ad):
        draw_topo_from_pos(pos, ad)
        plt.show()
def Traim_and_draw():
    train(train_datasets, EPOCH)
    generator.save(f'{PLT_SAVE_FILE}\\mse_generator.h5')
    discriminator.save(f'{PLT_SAVE_FILE}\\mse_discriminator.h5')
    draw_train_loss()
    draw_test_ad_score()
if __name__ == '__main__':
    Traim_and_draw()

