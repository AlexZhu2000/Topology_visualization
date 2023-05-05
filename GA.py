import numpy as np
import copy
# from layout_evaluate import topo_evaluate_single
import networkx as nx
import numpy as np
from My_Graph_Class import My_Graph
from nodes_connections_layout import drawboard_analyze
def reproduce_graph_from_pos_adj(pos, adj):
    '''
    recreate a my_graph from single pair (pos, adj)
    :param pos:
    :param adj:
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
    return my_graph
def Distance(p1, p2):
    '''

    :param p1:
    :param p2:
    :return:
    '''
    (x1, y1) = p1
    (x2, y2) = p2
    return pow((x1 - x2) ** 2 + (y1 - y2) ** 2, 0.5)
def D1(my_graph):
    '''
    根节点分散程度
    :param my_graph:
    :return:
    '''
    width, height = drawboard_analyze(my_graph)
    L = pow(width ** 2 + height ** 2, 0.5)
    MAX_Score = 2 * (width + height + L)
    DIS = 0.
    for root_cnt, root in enumerate(my_graph.root[:-1]):
        p1 = my_graph.pos[root]
        for goal_root in my_graph.root[root_cnt + 1:]:
            DIS += Distance(p1, my_graph.pos[goal_root])
    score = DIS / MAX_Score
    # goal_dis = 4. * (4 ** 2)
    # dis = 0
    # for root in my_graph.root:
    #     (x, y) = my_graph.pos[root]
    #     if x not in [0, 8]:
    #         print(x)
    #     dis += (x - 4) ** 2
    # score = dis / goal_dis

    return score
def D2(my_graph):
    '''
    求 变电站 出线开关 中间节点 (簇)的平均坐标，求四个簇之间的距离和
    :param my_graph:
    :return:
    '''
    width, height = drawboard_analyze(my_graph)
    L = pow(width ** 2 + height ** 2, 0.5)
    MAX_Score = 2 * (width + height + L)
    DIS = 0.
    mean_root_pos = []
    # 先求得每个变电站及其所属子节点（簇）的平均坐标
    for root_cnt, root in enumerate(my_graph.root):
        p1 = my_graph.pos[root]
        (total_x, total_y) = p1
        root_switch = np.argwhere(my_graph.adjacency_matrix[root] == 1).flatten()
        for switch_node in root_switch:
            (x, y) = my_graph.pos[switch_node]
            total_x = (total_x + x)
            total_y = (total_y + y)
        mean_x = total_x / (len(root_switch) + 1)
        mean_y = total_y / (len(root_switch) + 1)
        mean_pos = tuple((mean_x, mean_y))
        mean_root_pos.append(mean_pos)

    # 获得各个簇中心之间的距离之和
    for pos_cnt in range(len(mean_root_pos) - 1):
        for goal_cnt in range(pos_cnt + 1, len(mean_root_pos)):
            DIS += Distance(mean_root_pos[pos_cnt], mean_root_pos[goal_cnt])
    score = DIS / MAX_Score
    return score
# def D2(my_graph):
#     '''
#     根和其叶子的聚集程度
#     :param my_graph:
#     :return:
#     '''
#     width, height = drawboard_analyze(my_graph)
#     S = width * height
#     area = 0.
#     # print('_____画布大小：', S)
#     for root in my_graph.root:
#         '''
#         对每个根节点的叶子结点计算面积
#         '''
#         root_leaves = np.argwhere(my_graph.adjacency_matrix[root] == 1).flatten()
#         (min_x, min_y) = (max_x, max_y) = my_graph.pos[root]
#         for leaf in root_leaves:
#             (x, y) = my_graph.pos[leaf]
#             if max_x < x:
#                 max_x = x
#             if min_x > x:
#                 min_x = x
#             if max_y < y:
#                 max_y = y
#             if min_y > y:
#                 min_y = y
#         area1 = (max_x - min_x) * (max_y - min_y)
#         area += area1
#     score = S / area
#     # if score > 1:
#     #     my_graph.show_graph()
#     #     plt.show()
#     return score
def D3(my_graph):
    '''
    求所有连线的平均距离
    :param my_graph:
    :return:
    '''
    all_dis = 0
    for node_pair in my_graph.edge:
        (node1, node2) = node_pair
        pos1 = my_graph.pos[node1]
        pos2 = my_graph.pos[node2]
        all_dis += Distance(pos1, pos2)
    mean_dis = all_dis / len(my_graph.edge)
    D3_MAX_SCORE = 2
    score = D3_MAX_SCORE / mean_dis
    # 满分为1， 一般mean_dis 比最高得分距离要高，所以一般小于1
    return score
import matplotlib.pyplot as plt
def topo_evaluate_batch(pos, ad):
    '''
    训练过程中对批量数据评估分数
    :param pos: batch_size * 18 *2
    :param ad: batch_size * 18 * 19
    :return:
    '''
    S = []
    # pos = pos.numpy()
    ad = ad.numpy()
    for single_pos, single_ad in zip(pos, ad):
        my_graph = reproduce_graph_from_pos_adj(single_pos, single_ad)
        s = D3(my_graph)
        score1 = D1(my_graph)
        score2 = D2(my_graph)
        score3 = D3(my_graph)
        s = (score2 + score1 + score3)
        # s = score2
        # s = score1
        # S.append(s)
        # if s > 1:
        #     good_graph = reproduce_graph_from_pos_adj(single_pos, single_ad)
        #     plt.figure()
        #     good_graph.show_graph()
        #     plt.savefig(f'generator_training\\good_score_graph\\{score1}_{score2}.jpg')
        #     plt.close()
        S.append(s)
    S = np.array(S)
    return S
def topo_evaluate_batch_WITH123(pos, ad):
    '''
    训练过程中对批量数据评估分数
    :param pos: batch_size * 18 *2
    :param ad: batch_size * 18 * 19
    :return:
    '''
    S = []
    S1 = []
    S2 = []
    S3 = []
    # pos = pos.numpy()
    # ad = ad.numpy()
    for single_pos, single_ad in zip(pos, ad):
        my_graph = reproduce_graph_from_pos_adj(single_pos, single_ad)
        s = D3(my_graph)
        score1 = D1(my_graph)
        score2 = D2(my_graph)
        score3 = D3(my_graph)
        s = (score2 + score1 + score3)
        # s = score2
        # s = score1
        # S.append(s)
        # if s > 1:
        #     good_graph = reproduce_graph_from_pos_adj(single_pos, single_ad)
        #     plt.figure()
        #     good_graph.show_graph()
        #     plt.savefig(f'generator_training\\good_score_graph\\{score1}_{score2}.jpg')
        #     plt.close()
        S1.append(score1)
        S2.append(score2)
        S3.append(score3)
        S.append(s)
    S = np.array(S)
    S1 = np.array(S1)
    S2 = np.array(S2)
    S3 = np.array(S3)
    return S1, S2, S3, S
def topo_evaluate_single(pos, adj):

    my_graph = reproduce_graph_from_pos_adj(pos, adj)

    score1 = D1(my_graph)
    score2 = D2(my_graph)
    score3 = D3(my_graph)
    s = (score2 + score1 + score3)
    return s

def evaluate(points, adj):
    pos = []
    for point in points:
        pos.append((point.x, point.y))
    pos = np.array(pos)
    return topo_evaluate_single(pos, adj)

class Point(object):
    def __init__(self, idx, x, y, cls):
        self.idx = idx
        self.x = x
        self.y = y
        self.cls = cls


class Individual(object):
    def __init__(self, points, adj):
        self.points = points
        self.adj = adj
        self.fitness = self.cal_fitness()
    def cal_fitness(self):
        return evaluate(self.points, self.adj)

    def cross_individual(self, other):
        n_points = []
        part_index = len(self.points) // 2 + 1
        for point in self.points[: part_index]:
            n_points.append(point)
        for point, back_point in zip(other.points[part_index:], self.points[part_index: ]):
            duplicate = False
            for pre_point in n_points:
                if pre_point.x == point.x and pre_point.y == point.y:
                    duplicate = True
                    break
            if duplicate:
                n_points.append(back_point)
            else:
                n_points.append(point)

        return Individual(n_points, self.adj)

    def valid_point(self, point):
        for iter_point in self.points:
            if point.x == iter_point.x and point.y == iter_point.y:
                return False
        return True


class GA(object):
    def __init__(self,
                 n=100,
                 cross_rate=0.8,
                 mutate_rate=0.8,
                 mutate_num=1,
                 mutate_region=(5, 5),
                 work_region=(100, 100),
                 max_generations=100,
                 stop_optimize=0.001
                 ):
        # 族群规模
        self.n = n
        # 交叉率
        self.cross_rate = cross_rate
        # 变异率
        self.mutate_rate = mutate_rate
        self.mutate_num = mutate_num
        # 变异范围
        self.mutate_region = mutate_region
        # 地图范围
        self.work_region = work_region
        # 最大遗传代数
        self.max_generations = max_generations
        # 停止条件，优化数值低于stop_rate后停止遗传
        self.stop_optimize = stop_optimize

    def optimize(self, init_samples):
        population = init_samples
        max_fitness = 0
        for G in range(self.max_generations):
            # print(f'start execute {G} iteration')
            # 1. 按照适应度排序
            population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
            if len(population) == 0 or population[0].fitness - max_fitness < self.stop_optimize:
                break
            # 2. 执行交叉和变异
            next_population = self.do_cross_and_mutate(population)
            # 3. 执行选择
            population = self.do_select(population, next_population)
            # print(f'end execute {G} iteration, current best fitness is {population[0].fitness}')
        print(f'end optimize, best fitness is {population[0].fitness}')
        return population

    def do_cross_and_mutate(self, population):
        next_population = []
        for father in population:
            child = father
            if np.random.rand() < self.cross_rate:
                mother = population[np.random.randint(len(population))]
                child = father.cross_individual(mother)
            child = self.do_mutate(copy.deepcopy(child))
            next_population.append(child)
        return next_population

    def do_mutate(self, individual):
        rand_idx = np.random.randint(0, len(individual.points), self.mutate_num)
        for mutate_index in rand_idx:
            if np.random.rand() < self.mutate_rate:
                # mutate_index = np.random.randint(len(individual.points))
                mutate_point = individual.points[mutate_index]
                while True:
                    mutate_x = np.random.randint(-self.mutate_region[0], self.mutate_region[0])
                    mutate_y = np.random.randint(-self.mutate_region[1], self.mutate_region[1])
                    n_point = Point(mutate_point.idx, mutate_point.x + mutate_x, mutate_point.y + mutate_y,
                                    mutate_point.cls)
                    # 处理越界
                    if n_point.x < 0:
                        n_point.x = 0
                    if n_point.x >= self.work_region[0]:
                        n_point.x = self.work_region[0] - 1
                    if n_point.y < 0:
                        n_point.y = 0
                    if n_point.y >= self.work_region[1]:
                        n_point.y = self.work_region[1] - 1
                    # 处理重复点
                    if individual.valid_point(n_point):
                        individual.points[mutate_index] = n_point
                        break
        return individual

    def do_select(self, population, next_population):
        for index in range(len(population)):
            if population[index].fitness < next_population[index].fitness:
                population[index] = next_population[index]
        return population
import tensorflow as tf
from multi_select import onehot_to_xy_batch, onehot_to_xy_1
from chess_GAN import draw_topo_from_pos
import matplotlib.pyplot as plt
def yichuan(adj_i, n, adj, gen_model):


    real_test_adj = adj[adj_i]
    label_2 = real_test_adj[:, -1]
    init_sample = []
    for i in range(n):
        real_test_adj_3 = real_test_adj.reshape((1, 30, 31))
        label_3 = label_2.reshape((1, 30, 1))
        a = np.arange(1, 5).astype('float32')
        deta = np.random.choice(a, 1)
        test_adj = real_test_adj_3[:, :, :-1] + np.random.random((1, 30, 30)) / deta
        test_adj = tf.concat([test_adj, label_3], axis=-1)
        pos = gen_model(test_adj)

        pos = onehot_to_xy_batch(pos, real_test_adj_3)

        pos = np.squeeze(pos)

        points = []
        for node in range(30):
            point = Point(node, pos[node][0], pos[node][1], cls=label_2[node])
            points.append(point)
        sample = Individual(points, real_test_adj)
        init_sample.append(sample)
    ga = GA(n)
    populations =ga.optimize(init_sample)
    points = populations[0].points
    pos = []
    for point in points:
        pos.append([point.x, point.y])
    # np.save(f'RandomForest\\{adj_i}.npy', np.array(pos))
    pos = np.array(pos)
    # draw_topo_from_pos(pos, real_test_adj)
    # plt.show()
    # print(populations)
    return points, pos
def yichuan_npy():
    pos = np.load('RandomForest\\1.npy')
    adj = np.load('train_data\\3000_adj.npy')
    draw_topo_from_pos(pos, adj[0])
    plt.show()
if __name__ == '__main__':
    # n = 100
    # p = 20
    # ga = GA(n=n)
    # init_samples = [Individual([Point(idx=i, x=np.random.randint(0, 100), y=np.random.randint(0, 100), cls=1) for i in range(p)]) for j in range(n)]
    # populations = ga.optimize(init_samples)
    # print(populations)
    # yichuan_npy()
    yichuan(3, 10, )