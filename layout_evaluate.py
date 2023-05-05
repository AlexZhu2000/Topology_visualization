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
    adj = np.array(adj)
    nodes_label = adj[:, -1]
    ad = adj[:, :-1]
    # print(ad.dtype, ad.shape)
    # print(type(ad))
    ad = ad.astype(np.int32)
    graph = nx.from_numpy_matrix(ad)

    root = np.argwhere(nodes_label == 1).flatten()
    root_switch = np.argwhere(nodes_label == 2).flatten()
    middle = np.argwhere(nodes_label == 3).flatten()
    contact_switch = np.argwhere(nodes_label == 4).flatten()
    node_num = len(pos)
    dict_pos = dict.fromkeys(np.arange(node_num))
    for i in range(len(pos)):
        dict_pos[i] = tuple(pos[i])
    my_graph = My_Graph(np.arange(node_num), root, root_switch, middle, contact_switch, graph.edges, dict_pos)
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
def D4(my_graph):
    '''
    交叉点和拐点
    :param my_graph:
    :return: 
    '''

def topo_evaluate_batch(pos, ad):
    '''
    训练过程中对批量数据评估分数
    :param pos: batch_size * 18 *2
    :param ad: batch_size * 18 * 19
    :return:
    '''
    S = []
    # pos = pos.numpy()
    # ad = ad.numpy()
    for single_pos, single_ad in zip(pos, ad):
        new_ad = single_ad
        if single_ad[-1][-1] == -1:
            # 说明有padding的无效节点，需要删除
            new_ad = []
            padding_num = 0
            for node in single_ad:
                if node[-1] == -1:
                    padding_num += 1
                    continue
                new_ad.append(node)
            new_ad = np.array(new_ad)
            new_ad = new_ad[:, :-padding_num]
            new_ad = np.array(new_ad)
            # print(new_ad.shape)
        my_graph = reproduce_graph_from_pos_adj(single_pos, new_ad)
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
    S = S.astype(np.float32)
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
def topo_evaluate_single_123(pos, adj):

    my_graph = reproduce_graph_from_pos_adj(pos, adj)

    score1 = D1(my_graph)
    score2 = D2(my_graph)
    score3 = D3(my_graph)
    s = (score2 + score1 + score3)
    return s, score1, score2, score3