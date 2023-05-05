
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pickle
import os
import networkx as nx
import numpy as np
from My_Graph_Class import My_Graph
import nodes_connections_layout as Nodes_Connection
CHESS_W = 11
CHESS_H = 11
CHESS_NUM = CHESS_H * CHESS_W
ADJ50_path = '.npy'
CHESS50_onehot_path = '.npy'
def READ_json_file(filepath):
    '''
    从json文件中提取并构造图
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
def xy2onehot():
    '''
    从电网坐标数据转换为Onehot编码
    :return:
    '''
    all_pos = np.load(open(ADJ50_path, 'rb'))
    all_chessboard = []
    for i, pos in enumerate(all_pos):
        # 对每一个图pos，有一个chess_map
        chess_map = np.zeros((30, CHESS_NUM))
        for n in range(30):
            # 对每个点 得到位置onehot_n
            (x, y) = pos[n]
            x = int(x)
            y = int(y)
            if y > 8:
                print('chessboard is small...')
                return
            onehot_n = (8 - y) * 9 + x
            chess_map[n][onehot_n] = 1
        all_chessboard.append(chess_map)
    all_chessboard = np.array(all_chessboard)
    np.save(CHESS50_onehot_path, all_chessboard)
    return all_chessboard
def load_train_data():
    '''
    加载补齐后的数据
    :return:
    '''
    adj = np.load(ADJ50_path)
    chess_cate = np.load(CHESS50_onehot_path)
    return chess_cate.astype('float32'), adj.astype('float32')

PLT_SAVE_FILE = 'chessboard\\30_81\\0928_no_same_pos'
pretrain_evaluate_model = tf.keras.models.load_model('topo_evaluate_model\\D3_30_2_model.h5')
pretrain_evaluate_model.trainable = False
global POS, ADJACENCY
POS, ADJACENCY = load_train_data()
print(POS.shape)

BUFFER_SIZE = 3000
BATCH_SIZE = 32
train_datasets = tf.data.Dataset.from_tensor_slices((POS, ADJACENCY))
# train_datasets = train_datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generator_model():
    model = keras.Sequential()
    # (n, 50 * 51)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024,  use_bias=False))      # (n, 2500)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(1536, use_bias=False))                           # (n, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(2048, use_bias=False))  # (n, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Dense(30 * 81, use_bias=False))    # (n, 30)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((30, 81)))                                     # (n, 18, 2)

    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1024, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    #
    model.add(tf.keras.layers.Dense(512, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
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
def discriminator_loss(real_out, real_pos, real_ad, fake_out, fake_pos, fake_ad):
    # gan
    # real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    # fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    global DIS_LOSS, MSE_DIS_LOSS, WGAN_DIS_LOSS
    #wgan的尝试
    real_loss = -tf.reduce_mean(real_out)
    fake_loss = tf.reduce_mean(fake_out)
    # mse-gan
    # mse_real_loss = MSE(topo_evaluate_batch(real_pos, real_ad), 1.)
    # mse_fake_loss = MSE(topo_evaluate_batch(fake_pos, fake_ad), 0.)
    # fake_loss = MSE(topo_evaluate_batch(fake_pos, fake_ad), fake_out)
    # fake_loss = MSE(tf.zeros_like(fake_out), fake_out)
    # print('\ndiscriminator_loss:', real_loss + fake_loss, '...')
    # WGAN_DIS_LOSS = np.append(WGAN_DIS_LOSS, real_loss + fake_loss)
    # MSE_DIS_LOSS = np.append(MSE_DIS_LOSS, mse_real_loss +mse_fake_loss)
    DIS_LOSS = np.append(DIS_LOSS, real_loss + fake_loss)
    return real_loss + fake_loss
MSE = tf.keras.losses.MeanSquaredError()

def generator_loss(fake_out, constrain_pos, ad, real_pos):
    global GEN_LOSS, SCORE_LOSS, MLP_LOSS, MSE_GEN_LOSS, WGAN_GEN_LOSS

    # gan
    # loss1 = cross_entropy(tf.ones_like(fake_out), fake_out)
    # mse_loss = cross_entropy(pos, real_pos)
    # wgan
    loss1 = -tf.reduce_mean(fake_out)
    # WGAN_GEN_LOSS = np.append(WGAN_GEN_LOSS, loss1)
    # lossx = MSE(real_pos[:, :, 0], pos[:, :, 0])
    # lossy = MSE(real_pos[:, :, 1], pos[:, :, 1])
    # mse-gan
    pos_xy = constrain_onehot_to_xy_batch(constrain_pos, ad)
    test_x = tf.concat([ad, pos_xy], axis=-1)
    s = pretrain_evaluate_model(test_x)
    mse_loss = MSE(pretrain_evaluate_model(test_x, training=False), 3 * tf.ones_like(s))
    # print('\t \t \t \tgenerator_loss:', loss1 ,'\t','Evaluate loss:',loss2,  '...\n')
    GEN_LOSS = np.append(GEN_LOSS, loss1 + mse_loss)
    WGAN_GEN_LOSS = np.append(GEN_LOSS, loss1)
    # MSE_GEN_LOSS = np.append(MSE_GEN_LOSS, mse_gen_loss)
    MLP_LOSS = np.append(MLP_LOSS, mse_loss)



    # SCORE_LOSS = np.append(SCORE_LOSS, loss2)
    return 0.2 * loss1 + 0.8 * mse_loss

discriminator_opt = tf.keras.optimizers.Adam(1e-4)
generator_opt = tf.keras.optimizers.Adam(1e-4)
# discriminator_opt = tf.keras.optimizers.RMSprop(0.001)
# generator_opt = tf.keras.optimizers.RMSprop(0.0001)
generator = generator_model()
discriminator = discriminator_model()
def gen_pos_constrain(gen_pos, adj):
    '''
    将根节点对应的备选位置对应值保留，其余位置概率置0
    :param gen_pos: 32 * 30 * 81
    :return:
    '''
    np_gen_pos = np.array(gen_pos)
    np_adj = np.array(adj)

    new_np_gen_pos = []
    for i in range(gen_pos.shape[0]):
        single_pos, single_adj = np_gen_pos[i], np_adj[i]
        new_single_pos = []
        for node, onehot in enumerate(single_pos):
            if single_adj[node][-1] == 1:
                # 如果该node为根节点
                onehot[OTHER_INDEX] = 0
                new_single_pos.append(onehot)
            else:
                onehot[ROOT_INDEX] = 0
                new_single_pos.append(onehot)
        # new_single_pos = tf.constant(new_single_pos).reshape((1, 30, 81))
        # print(new_single_pos)
        # new_gen_pos = tf.concat([new_gen_pos, new_single_pos], axis=0)
        # tf.tensor_scatter_nd_update(gen_pos, [i], new_single_pos)
        new_np_gen_pos.append(new_single_pos)

    return tf.constant(np.array(new_np_gen_pos))
def constrain_linear(gen_pos, adj):
    mask_batch = []
    for i in range(adj.shape[0]):
        single_mask = []
        for node, adj_label in enumerate(adj[i]):
            if adj_label[-1] == 1:
                #如果为根节点，对应Mask设置为1
                temp = np.zeros(81)
                temp[ROOT_INDEX] = 1
                single_mask.append(temp)
            elif adj_label[-1] == 2:
                temp = np.zeros(81)
                temp[ROOT_SWITCH_INDEX] = 1
                single_mask.append(temp)
            elif adj_label[-1] == 3:
                temp = np.zeros(81)
                temp[MIDDLE_INDEX] = 1
                single_mask.append(temp)
            elif adj_label[-1] == 4:
                temp = np.zeros(81)
                temp[CONTACT_INDEX] = 1
                single_mask.append(temp)
            else:
                print('WRONG NODE LABEL IN TRAIN CONSTRAIN....')
                temp = np.ones(81)
                temp[ROOT_INDEX] = 0
                temp[ROOT_SWITCH_INDEX] = 0
                single_mask.append(temp)
        mask_batch.append(single_mask)
    mask_batch = np.array(mask_batch).astype('float32')
    mask_batch = tf.constant(mask_batch)

    return mask_batch
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
        real_condition_pair = tf.concat([pos, ad[:, :, :-1]], 2)
        real_out = discriminator(real_condition_pair, training=True)
        #
        gen_position = generator(ad, training=True)
        mask = constrain_linear(gen_position, ad)
        constrain_pos = tf.multiply(gen_position, mask)
        fake_condition_pair = tf.concat([constrain_pos, ad[:, :, :-1]], 2)
        fake_out = discriminator(fake_condition_pair, training=True)
        fake_pos = constrain_pos

        # pos = tf.expand_dims(pos, -1)
        # real_pos = tf.concat([ad, pos],axis=-1)
        # real_out = discriminator(real_pos, training=True)            #real_pos (30 * 32)
        #
        # gen_position = generator(ad, training=True)                 # gen_position (30)
        # fake_pos = tf.concat([ad, tf.expand_dims(gen_position, -1)], axis=-1)
        # fake_out = discriminator(fake_pos, training=True)
        # real_out = discriminator(pos, training=True)
        # gen_pos = generator(ad, training=True)
        # fake_pos = gen_pos
        # fake_out = discriminator(fake_pos, training=True)
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
from layout_evaluate import D1, D2,reproduce_graph_from_pos_adj, D3
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
def constrain_onehot_to_xy_batch(constrain_pos, adj):
    output = []
    constrain_pos = np.array(constrain_pos)
    for pos, ad in zip(constrain_pos, adj):
        single_pos = constrain_onehot_to_xy(pos, ad)
        output.append(single_pos)
    return output
def constrain_onehot_to_xy(all_onehot, adj):
    all_output = tf.nn.softmax(all_onehot)
    all_output = np.array(all_output)

    pos = []
    for node, onehot in enumerate(all_output):
        index = np.argmax(onehot)
        y = 8 - int(index / 9)
        x = index - (8 - y) * 9
        while (x, y) in pos:
            onehot[index] = 0
            index = np.argmax(onehot)
            y = 8 - int((index) / 9)
            x = index - (8 - y) * 9
        pos.append((x, y))
    return np.array(pos)
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
def cate_to_xy(cate):
    '''

    :param cate:
    :return:
    '''
def generate_plot_position(gen_model, generate_ad, n):
    # pre_images = gen_model(test_noise, trainable=False)
    generate_pos = gen_model(generate_ad)
    # print(f'-------------\n{generate_pos}\n--------------\n')
    fig = plt.figure(figsize=(10, 10))
    # all_graph = []
    for i in range(generate_pos.shape[0]):
        plt.subplot(2, 2, i + 1)
        # plt.imshow((generate_pos[i, :, :] + 1) / 2, cmap='gray')
        # 这里是因为我们使用tanh激活函数之后会将结果限制在-1到1之间，而我们需要将其转化到0到1之间。
        pos = onehot_to_xy(generate_pos[i], generate_ad[i])

        draw_topo_from_pos(pos, generate_ad[i])
        my_graph = reproduce_graph_from_pos_adj(pos, generate_ad[i])
        d1 = D1(my_graph)
        d2 = D2(my_graph)
        d3 = D3(my_graph)
        AD[i] = np.append(AD[i], d1)
        AD_SCORE2[i] = np.append(AD_SCORE2[i], d2)
        AD_SCORE3[i] = np.append(AD_SCORE3[i], d3)
        FULL_SCORE[i] = np.append(FULL_SCORE[i], d1 + d2 + d3)
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
        plt.plot(AD_SCORE2[i], color='b', linestyle='--', label='D2')
        plt.plot(AD_SCORE3[i], color='g', linestyle='-.', label='D3')
        plt.plot(FULL_SCORE[i], color='k', linestyle=':', label='all')
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
        # generate_plot_position(generator, generate_ad, i)
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

