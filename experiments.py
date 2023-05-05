
from multi_select import *
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

def draw_with_generator(model, test_adj):
    '''
    test_adj:(x, 30, 31)
    :param model:"xxxx.h5"
    :param test_adj:
    :return:
    '''
    pos = model(test_adj)
    for p, a in zip(pos, test_adj):
        draw_topo_from_pos(p, a)
        plt.show()
from layout_evaluate import topo_evaluate_batch_WITH123, topo_evaluate_batch
import nodes_connections_layout as Nodes_Connection
from GA import yichuan
def mse_cgan_train_generate_compare():
    adj = np.load('train_data\\3000_adj.npy')
    pos_onehot = np.load('train_data\\3000_chessboard.npy')
    posxy = onehot_to_xy_batch(pos_onehot, adj)
    s1, s2, s3, s = topo_evaluate_batch_WITH123(posxy, adj)

    gen_pos = []
    gen_model = tf.keras.models.load_model(r'chessboard\30_81\0925_all_D_1\mse_generator.h5')
    for i in range(3000):
        best_point, single_posxy = yichuan(i, 10, adj, gen_model)
        gen_pos.append(single_posxy)
        print(f'xxxxxxxxxxxxxxxx{i}xxxxxxxxxxxxxxxxxxx')
    # mse_cgan_model = tf.keras.models.load_model(r'chessboard\30_81\0925_all_D_1\mse_generator.h5')
    # gen_pos_onehot = mse_cgan_model(adj)
    # gen_posxy = onehot_to_xy_batch(gen_pos_onehot, adj)
    gen_s1, gen_s2, gen_s3, gen_s = topo_evaluate_batch_WITH123(gen_pos, adj)

    np.save('Experiment_data\\train_gen_score\\3000_train_score.npy', s)
    np.save('Experiment_data\\train_gen_score\\3000_train_D1_score.npy', s1)
    np.save('Experiment_data\\train_gen_score\\3000_train_D2_score.npy', s2)
    np.save('Experiment_data\\train_gen_score\\3000_train_D3_score.npy', s3)

    np.save('Experiment_data\\train_gen_score\\3000_gen_score.npy', gen_s)
    np.save('Experiment_data\\train_gen_score\\3000_gen_D1_score.npy', gen_s1)
    np.save('Experiment_data\\train_gen_score\\3000_gen_D2_score.npy', gen_s2)
    np.save('Experiment_data\\train_gen_score\\3000_gen_D3_score.npy', gen_s3)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.plot(s, color='r', label='train')
    plt.plot(gen_s, color='b', linestyle='-', label='gen')
    plt.title('分数变化')
    plt.xlabel('轮次')
    plt.ylabel('分数')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(s1, color='r', label='train')
    plt.plot(gen_s1, color='b', linestyle='-', label='gen')
    plt.title('分数变化')
    plt.xlabel('轮次')
    plt.ylabel('分数')
    plt.legend()
    plt.show()
    # plt.savefig(os.path.join(PLT_SAVE_FILE, f'adj_score_{i}.png'))
def diff_score_data():
    mse_gan_path = r'chessboard\30_81\0925_all_D_1\mse_generator.h5'
    gan_path = 'chessboard\\gan\\gan_30_2_3\\mse_generator.h5'
    wgan_path = r'generator_training\\0904_shuffle_wgan\\mse_generator.h5'
    mlp_path = 'chessboard\\mlp\\0907_mlp\\drop_out_l2_mlp.h5'

    mse_model = tf.keras.models.load_model(mse_gan_path)
    mlp_model = tf.keras.models.load_model(mlp_path)
    gan_model = tf.keras.models.load_model(gan_path)
    wgan_model = tf.keras.models.load_model(wgan_path)

    adj3000 = np.load('train_data\\3000_adj.npy')
    posxy3000 = np.load('train_data\\3000_pos.npy')
    # 2, 3, 5
    adj_index = 3
    test_adj3 = adj3000[adj_index].reshape((1, 30 ,31))
    # for i in range(50):
    #     score = topo_evaluate_single(posxy3000[i], adj3000[i])
    #     if score > 2.5:
    #         print('adj:', i, '\tscore:', score)

    all_model = [gan_model, wgan_model, mlp_model, mse_model]

    # for model in all_model:
    #     pos = model(test_adj3)
    #     if model == mse_model:
    #         pos = onehot_to_xy_batch(pos, test_adj3)
    #     draw_topo_from_pos(np.squeeze(pos), adj3000[adj_index])
    #     plt.show()
    #     score = topo_evaluate_batch(pos, test_adj3)
    #     print(score)
    adj = adj3000[adj_index]

    path = 'Experiment_data\\score_com'
    pos1_5 = mse_model(test_adj3)
    posxy1_5 = onehot_to_xy_batch(pos1_5, test_adj3)
    score = topo_evaluate_batch(posxy1_5, test_adj3)
    G = nx.from_numpy_matrix(adj3000[adj_index][:, :-1])

    edge_switch_to_json(G, path, score)
    pos_switch_to_json(score, adj3000[adj_index][:, -1], np.squeeze(posxy1_5), path)

    # xunlianshuju
    posxy = posxy3000[adj_index]
    score2 = topo_evaluate_single(posxy, adj)
    pos_switch_to_json(score2, adj[:, -1],posxy, path)

def edge_switch_to_json(G, path ):
    edges = []
    for edge in G.edges:
        a, b = edge[0], edge[1]
        edges_josn = {
            'from_node': a,
            'to_node': b
        }
        edges.append(edges_josn)

    edges_json = {'edges': edges}
    save_path = path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(f'{save_path}\\edges.json', "w") as f:
        json.dump(edges_json, f)
        print('save edges file.......')
    return True
def pos_switch_to_json(score, label, node_pos, path):
    # print(self.G.edges)
    nodes = []
    for node in range(18):
        node_dict = {
            'id':int(node),
            'pos':list(node_pos[node].tolist()),
            'label':str(int(label[node]))
        }
        nodes.append(node_dict)

    all_json = {
        'nodes':nodes,
    }
    save_path = path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(f'{save_path}\\{score}.json', "w") as f:
        json.dump(all_json, f)

    return True

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
    length = len(pos)
    dict_pos = dict.fromkeys(np.arange(length))
    for i in range(length):
        dict_pos[i] = tuple(pos[i])
    my_graph = My_Graph(np.arange(length), root, root_switch, middle, contact_switch, graph.edges, dict_pos)
    # Nodes_Connection.draw_topo_from_graph(my_graph)
    my_graph.show_graph()
def all_model_compare():
    gan_path = 'chessboard\\gan\\gan_30_2_3\\mse_generator.h5'
    wgan_path = 'chessboard\\wgan\\wgan_30_2_discrete_pos_plt_6_lossxy_test\\mse_generator.h5'
    wgan_path = r'generator_training\\0904_shuffle_wgan\\mse_generator.h5'
    mlp_path = 'chessboard\\mlp\\0907_mlp\\drop_out_l2_mlp.h5'
    mse_gan_path = r'chessboard\30_81\0925_all_D_1\mse_generator.h5'
    cgan_path = r'chessboard\\condition_gan\\0915_1\\mse_generator.h5'

    gan_model = tf.keras.models.load_model(gan_path)
    wgan_model = tf.keras.models.load_model(wgan_path)
    mlp_model = tf.keras.models.load_model(mlp_path)
    mse_model = tf.keras.models.load_model(mse_gan_path)
    cgan_model = tf.keras.models.load_model(cgan_path)

    all_model = [gan_model, wgan_model, mlp_model, mse_model]
    all_model_name = ['gan', 'wgan', 'mlp', 'mse']
    all_model_dict = dict(zip(all_model_name, all_model))
    print(all_model_dict)
    adj = np.load('train_data\\3000_adj.npy')
    test_adj = adj[0].reshape((1, 30, 31))

    for name, model in all_model_dict.items():
        print(model.summary())
        pos = model(test_adj)
        if model == mse_model:
            pos = onehot_to_xy_batch(pos, test_adj)
        for p, a in zip(pos, test_adj):
            draw_topo_from_pos(p, a)
            plt.title(name)
            plt.show()
        test_adj_ = tf.convert_to_tensor(test_adj)
        s = topo_evaluate_batch(pos, test_adj_)
        print(s)
def train_pos_visualise():
    adj = np.load('train_data\\3000_adj.npy')
    chessboard_pos = np.load('train_data\\3000_chessboard.npy')
    posxy = onehot_to_xy_batch(chessboard_pos, adj)
    i = 0
    for pos, a in zip(posxy, adj):
        draw_topo_from_pos(pos, a)
        plt.savefig(f'train_topo_pics\\3000_chessboard_reduction\\{i}.jpg')
        # plt.show(block=False)
        i += 1
import json
gen_model = tf.keras.models.load_model(r'chessboard\30_81\0925_all_D_1\mse_generator.h5')
class Gen():
    def __init__(self, node_size, adj, adj_num, gen_num, adj_path):
        self.nodes = np.arange(node_size)
        self.adj_num = adj_num
        self.adj = adj
        self.ad = adj[:, :-1]
        self.label = adj[:, -1]
        self.gen_num = gen_num
        self.adj_path = adj_path
        self.G = nx.from_numpy_matrix(self.ad)
    def gen_pos_with_noixe(self):
        real_test_adj = self.adj
        label_2 = real_test_adj[:, -1]
        self.edge_switch_to_json()
        for i in range(self.gen_num):
            real_test_adj_3 = real_test_adj.reshape((1, 30, 31))
            label_3 = label_2.reshape((1, 30, 1))
            a = np.arange(1, 5).astype('float32')
            deta = np.random.choice(a, 1)
            test_adj = real_test_adj_3[:, :, :-1] + np.random.random((1, 30, 30)) / deta
            test_adj = tf.concat([test_adj, label_3], axis=-1)
            pos = gen_model(test_adj)
            pos = onehot_to_xy_batch(pos, real_test_adj_3)
            pos = np.squeeze(pos)
            self.pos_switch_to_json(i, pos)

    def edge_switch_to_json(self):
        edges = []
        for edge in self.G.edges:
            a, b = edge[0], edge[1]
            edges_josn = {
                'from_node': a,
                'to_node': b
            }
            edges.append(edges_josn)

        edges_json = {'edges': edges}
        save_path = self.adj_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(f'{save_path}\\edges.json', "w") as f:
            json.dump(edges_json, f)
            print('save edges file.......')
        return True
    def pos_switch_to_json(self, index, node_pos):
        # print(self.G.edges)
        nodes = []
        for node in self.nodes:
            node_dict = {
                'id':int(node),
                'pos':list(node_pos[node].tolist()),
                'label':str(int(self.label[node]))
            }
            nodes.append(node_dict)

        all_json = {
            'nodes':nodes,
        }
        save_path = self.adj_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(f'{save_path}\\{self.adj_num}_{index}.json', "w") as f:
            json.dump(all_json, f)

        return True
def shiyan_18():
    # adj = np.load(r'F:\zzh\pycharm_program\Graph\graph_data\18-19adjacency.npy')
    # pos = np.load(r"F:\zzh\pycharm_program\Graph\graph_data\18-19pos.npy")
    # print(type(pos))
    # # adj = np.load('train_data\\3000_adj.npy')
    # # pos = np.load('train_data\\3000_pos.npy')
    # print(adj.shape, pos.shape)
    # a = topo_evaluate_single(pos[0], adj[0])
    # g = nx.from_numpy_matrix(adj[0][:, :-1])
    # nx.draw_networkx(g, pos=pos[0],  with_labels=False)
    # plt.show()
    # print(a)

    root = [0, 1]
    root_switch = [2, 3, 4, 5, 6, 7]
    middle = [8, 9, 10, 11, 12, 13]
    contact = [14, 15, 16, 17]
    edges = [(0, 2), (0, 3), (0, 4),
             (1, 5), (1, 6), (1, 7),
             (2, 8), (3, 9), (4, 10), (5, 11), (6, 12), (7, 13),
             (8, 14), (14, 11),
             (8, 15), (15, 12),
             (9, 16), (16, 12),
             (10, 17), (17, 13)]
    pos2_768 = [(0, 1), (6, 1), (1, 3), (1, 1), (1, 0),
           (5, 3), (5, 1), (5, 0), (2, 3), (2, 1),
           (2, 0), (4, 3), (4, 1), (4, 0), (3, 3),
           (3, 2), (3, 1), (3, 0)]

    pos_2 = [(0, 1), (6, 1), (5, 3), (5, 1), (5, 0),
            (1, 3), (1, 1), (1, 0),  (2, 1), (2, 0),
             (4, 3), (4, 1), (2, 3), (4, 0), (3, 3),
            (3, 2), (3, 1), (3, 0)]

    pos_1 = [(0, 1),  (1, 3),  (1, 0),
           (5, 3), (5, 1), (5, 0),(1, 1), (2, 3), (2, 1),
           (2, 0), (4, 3),(6, 1), (4, 1), (4, 0), (3, 3),
           (3, 2), (3, 1), (3, 0)]
    G = nx.Graph()
    G.add_edges_from(edges)
    label = [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]

    pos_switch_to_json(2.8462, label, np.array(pos2_768), 'Experiment_data\\18_pos')
    pos_switch_to_json(1.9778, label, np.array(pos_2), 'Experiment_data\\18_pos')
    pos_switch_to_json(0.9759, label, np.array(pos_1), 'Experiment_data\\18_pos')
    edge_switch_to_json(G, 'Experiment_data\\18_pos')


    # adj = np.array(nx.adjacency_matrix(G).todense())
    #
    #
    # label = np.array(label).T.reshape((18 ,1))
    # new_adj = np.concatenate((adj, label), axis=-1)
    # np.save('Experiment_data\\18_adj.npy', new_adj)
    # nx.draw_networkx(G, pos=pos2_768)
    # plt.show()
    #
    # ##############
    # my_graph = My_Graph(np.arange(18), root, root_switch ,middle, contact, edges, pos2_768)
    # my_graph.show_graph()
    # plt.show()
    #
    # #############]
    # my_graph = My_Graph(np.arange(18), root, root_switch, middle, contact, edges, pos_2)
    # my_graph.show_graph()
    # plt.show()
    #
    # ############
    # my_graph = My_Graph(np.arange(18), root, root_switch, middle, contact, edges, pos_1)
    # my_graph.show_graph()
    # plt.show()
    # a, s1, s2, s3 = topo_evaluate_single_123(pos_2, new_adj)
    # print(a, s1 ,s2, s3)
if __name__ == '__main__':
    adj = np.load('train_data\\3000_adj.npy')
    #
    # for i, a in enumerate(adj[:500]):
    #     gen = Gen(30, a, i, 1000, f'Experiment_data\\yichuan_pos\\adj{i}')
    #     print(f'xxxxxxxxxxxxxxxxxxx this is the {i} adj xxxxxxxxxxxxxxxxx')
    #     gen.gen_pos_with_noixe()
    # diff_score_data()
    # mse_gan_path = r'chessboard\30_81\0925_all_D_1\mse_generator.h5'
    # mse_model = tf.keras.models.load_model(mse_gan_path)
    # best_point, single_posxy = yichuan(3, 10, adj, mse_model)
    # path = 'Experiment_data\\score_com'
    # print(best_point)
    # pos_switch_to_json(1.749975, adj[3][:, -1], single_posxy, path)
    #
    # print(best_point)
    shiyan_18()