
import PATH
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import tensorflow as tf
import copy
from My_Graph_Class import My_Graph
import nodes_connections_layout as Nodes_Connection

def READ_json_file(filepath):
    '''

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
def Find_contact_nodes(edges, node):
    '''
    从联络开关和边，得到与之相连的两个点
    :param edge:
    :param node:
    :return:
    '''
    label = 0
    for (x, y) in edges:
        if x == node:
            a = y
            label += 1
        if y == node:
            b = x
            label += 1
        if label == 2:
            return a, b
    return a, b
def Generate_topology_arange_position(node_num, root, root_switch_tree, middle_tree, contact_switch, edges):
    '''
    arange position for all labels nodes
    :return:
    '''
    pos = dict.fromkeys(np.arange(node_num))
    tree_length = []
    for i in range(4):
        tree_length.append(len(root_switch_tree[i]))
    # 安排变电站根节点坐标
    pos[root[0]] = (0, int(tree_length[0] / 2))
    pos[root[1]] = (0, tree_length[0] + int(tree_length[1] / 2))
    pos[root[2]] = (8, int(tree_length[2] / 2))
    pos[root[3]] = (8, tree_length[2] + int(tree_length[3] / 2))
    # 安排出线开关和中间节点的位置
    for i in range(4):
        for j in range(tree_length[i]):
            if i == 0:
                pos[root_switch_tree[i][j]] = (1, j)
                pos[middle_tree[i][j]] = (2, j)
            if i == 1:
                pos[root_switch_tree[i][j]] = (1, j + tree_length[0])
                pos[middle_tree[i][j]] = (2, j + tree_length[0])
            if i == 2:
                pos[root_switch_tree[i][j]] = (7, j)
                pos[middle_tree[i][j]] = (6, j)
            if i == 3:
                pos[root_switch_tree[i][j]] = (7, j + tree_length[2])
                pos[middle_tree[i][j]] = (6, j + tree_length[2])
    # 安排联络开关的位置
    left_3_y = []
    center_4_y = []
    right_5_y = []
    for node in contact_switch:
        a, b = Find_contact_nodes(edges, node)
        # 找到联络开关联络的两个节点
        for i in range(4):
            if a in middle_tree[i]:
                a_label = i
            if b in middle_tree[i]:
                b_label = i
        # print('the contact switch nodes area are:', a_label, b_label)
        if a_label < 2 and b_label < 2:
            # 如果都在左侧
            node_y = (pos[a][1] + pos[b][1]) / 2
            if node_y not in left_3_y:
                left_3_y.append(node_y)
            else:
                node_y -= 0.75
                left_3_y.append(node_y)
            pos[node] = (3, node_y)
            # print('-------------in the left-----------')
        elif a_label > 1 and b_label > 1:
            #如果都在右侧
            node_y = (pos[a][1] + pos[b][1]) / 2
            if node_y not in right_5_y:
                right_5_y.append(node_y)
            else:
                node_y -= 0.75
                right_5_y.append(node_y)
            pos[node] = (5, node_y)
        else:
            # 在两边，即对侧相连
            node_y = (pos[a][1] + pos[b][1]) / 2
            if node_y not in center_4_y:
                center_4_y.append(node_y)
            else:
                node_y -= 0.75
                center_4_y.append(node_y)
            pos[node] = (4, node_y)
    # print((pos.keys()))
    return pos
def Generate_topology(node_num):
    '''
    root \ root_switch \ middle_nodes \ contact_switch_nodes

    root -- root_switch_tree -- middle_tree
    generate true topology layout with node_num
    :param node_num:  30
    :return:
    '''
    nodes = np.arange(node_num)
    # 根节点
    root = np.random.choice(nodes, 4, replace=False)
    curr_nodes = np.array(list(set(nodes) - set(root)))

    # 出线开关
    Root_Switch_num = np.random.randint(7, 11)
    root_switch = np.random.choice(curr_nodes, Root_Switch_num, replace=False)
    curr_nodes = np.array(list(set(curr_nodes) - set(root_switch)))

    # 中间节点 和出线开关数量一致
    Middle_nodes_num = Root_Switch_num
    # print('middle_nodes_num:', Middle_nodes_num)
    middle_nodes = np.random.choice(curr_nodes, Middle_nodes_num, replace=False)
    curr_nodes = np.array(list(set(curr_nodes) - set(middle_nodes)))
    # 联络开关
    contact_switch = curr_nodes
    contact_switch_nodes = contact_switch
    Contact_Switch_num = node_num - 4 - Root_Switch_num - Middle_nodes_num
    # print('contact_switch_num:', Contact_Switch_num)
    # 安排出线开关
    root_switch_bag = root_switch
    random.shuffle(root_switch_bag)
    tree1 = np.random.choice(root_switch_bag, random.randint(1, len(root_switch_bag) - 3), replace=False)
    root_switch_bag = list(set(root_switch_bag) - set(tree1))
    tree2 = np.random.choice(root_switch_bag, random.randint(1, len(root_switch_bag) - 2), replace=False)
    root_switch_bag = list(set(root_switch_bag) - set(tree2))
    tree3 = np.random.choice(root_switch_bag, random.randint(1, len(root_switch_bag) - 1), replace=False)
    root_switch_bag = list(set(root_switch_bag) - set(tree3))
    tree4 = root_switch_bag
    root_switch_tree = [tree1, tree2, tree3, tree4]

    middle_bag = middle_nodes
    random.shuffle(middle_bag)
    middle = [0, 0, 0, 0]

    # 安排中间节点
    for i in range(4):

        middle[i] = np.random.choice(middle_bag, len(root_switch_tree[i]), replace=False)
        middle_bag = list(set(middle_bag) - set(middle[i]))
    middle_tree = middle
    # middle1 = np.random.choice(middle_bag, len(tree1), replace=False)
    # middle_bag = list(set(middle_bag) - set(middle1))
    # middle2 = np.random.choice(middle_bag, len(tree2), replace=False)
    # middle_bag = list(set(middle_bag) - set(middle1))
    # middle3 = np.random.choice(middle_bag, len(tree3), replace=False)
    # middle4 = np.random.choice(middle_nodes, len(tree4), replace=False)

    edge = []
    # 添加除联络开关之外的边
    for i in range(4):
        for j in range(len(root_switch_tree[i])):
            edge.append((root[i], root_switch_tree[i][j]))
            edge.append((root_switch_tree[i][j], middle_tree[i][j]))
    # 添加联络开关相关的边
    middle_bag = middle_nodes
    random.shuffle(middle_bag)
    random.shuffle(contact_switch)
    if len(middle_bag) / 2 >= Contact_Switch_num:
        # 如果中间节点两两相连足够分配给 联络开关，则直接分配
        # use_contact_switch_num = len(middle_bag) / 2
        for i in range(Contact_Switch_num):
            edge.append((middle_bag[2 * i], contact_switch[i]))
            edge.append((contact_switch[i], middle_bag[2 * i + 1]))
    else:
        # 如果中间节点两两相连，不够联络开关用，则
        use_contact_switch_num = int(len(middle_bag) / 2)
        for i in range(use_contact_switch_num):
            edge.append((middle_bag[2 * i], contact_switch[i]))
            edge.append((contact_switch[i], middle_bag[2 * i + 1]))
        more_pair_num = Contact_Switch_num - use_contact_switch_num
        if more_pair_num > Middle_nodes_num:
            # 如果还需要的连接数大于中间节点，防止重复边，需要重新设计
            for i in range(Contact_Switch_num - use_contact_switch_num):
                if i >= Middle_nodes_num:
                    index = i % Middle_nodes_num
                    edge.append((middle_bag[index], contact_switch[i + use_contact_switch_num]))
                    edge.append((contact_switch[i + use_contact_switch_num], middle_bag[(index + 3) % Middle_nodes_num]))
                else:
                    edge.append((middle_bag[i], contact_switch[i + use_contact_switch_num]))
                    edge.append((contact_switch[i + use_contact_switch_num], middle_bag[(i + 2) % Middle_nodes_num]))
        else:
            for i in range(Contact_Switch_num - use_contact_switch_num):
                edge.append((middle_bag[i], contact_switch[i + use_contact_switch_num]))
                edge.append((contact_switch[i + use_contact_switch_num], middle_bag[(i + 3) % Middle_nodes_num]))
    right_edge = 2 * Root_Switch_num + 2 * Contact_Switch_num
    if right_edge == len(edge):
        print('the edges are right...')
    else:
        print('WRONG')

    # 为节点安排坐标
    pos = Generate_topology_arange_position(node_num, root, root_switch_tree, middle_tree, contact_switch, edge)

    my_graph = My_Graph(nodes, root, root_switch, middle_nodes, contact_switch_nodes, edge, pos)
    graph_copy = copy.deepcopy(my_graph)
    Nodes_Connection.draw_topo_from_graph(graph_copy)
    return my_graph
def Generate_graph_save_train_data(n):
    all_graph = []
    all_adj = []
    all_pos = []
    for i in range(n):
        graph = Generate_topology(30)

        pos = list(graph.pos.values())
        #
        pos = np.array(pos)
        print(pos.shape, type(pos))
        ad = graph.adjacency_matrix
        root = graph.root
        root_switch = graph.root_switch
        middle = graph.middle
        contact_switch = graph.contact_switch
        label = np.zeros(30)
        label[root] = 1
        label[root_switch] = 2
        label[middle] = 3
        label[contact_switch] = 4
        label = tf.expand_dims(label, axis=1)
        #
        adj = tf.concat([ad, label], 1).numpy()

        all_graph.append(graph)
        all_adj.append(adj)
        all_pos.append(pos)

        plt.savefig(f'train_topo_pics\\{n}\\{i}.jpg')
        plt.show()
    outfile = open(f'train_topo_pics\\{n}\\{n}_graph.pkl', 'wb')
    OK = pickle.dump(all_graph, outfile)
    all_adj = np.array(all_adj).astype('float32')
    all_pos = np.array(all_pos).astype('float32')
    np.save(f'train_topo_pics\\{n}\\{n}_graph_adj.npy', all_adj)
    np.save(f'train_topo_pics\\{n}\\{n}_graph_pos.npy', all_pos)
    print(all_pos.shape, all_adj.shape)
if __name__ == '__main__':
    # g = READ_json_file(PATH.JSON_PATH + '1.json')
    # print(g.edges(data=True))
    Generate_graph_save_train_data(100)
    # for i in range(10):
    #     graph = Generate_topology(30)
    #     plt.show()