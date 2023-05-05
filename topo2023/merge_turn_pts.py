import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
def SAVE_JSON(dict, path):
    '''

    :return:
    '''
    print(path)
    with open(path, 'w') as fp:
        json.dump(dict, fp)
def draw_graph_with_pos(G, ori_nodes, save_path = None):
    '''
    和 merge_one_graph绑定
    :param G:
    :param pos:
    :return:
    '''
    G = G.to_undirected()
    nx_pos = {}
    count = 0
    for node in ori_nodes:
        nx_pos[node['id']] = node['pos']
        count += 1
    nx.draw_networkx(G, with_labels=False, node_size=5, pos=nx_pos)
    plt.savefig('train_data/' + save_path +'/merge_graph')
    plt.show()
def check_json():
    path = 'train_data/tp50/1000_2.json'
    with open(path, 'r') as fp:
        graph = json.load(fp)
    nodes = graph['nodes']
    edges = graph['edges']
    type5_num = 0
    contact_nodes = []
    pos = {}
    for node in nodes:
        if node['type'] == 2:
            # nodes.remove(node)
            type5_num += 1
            contact_nodes.append(node['id'])
            pos[node['id']] = node['pos']
    G = nx.Graph()
    G.add_nodes_from(contact_nodes)
    nx.draw_networkx(G, pos = pos)
    plt.show()
    print('nodes num:', len(nodes), 'type2_nodes num:', type5_num)
def merge_one_graph(path):
    '''
    合并单个json文件中的拐点
    :param path:
    :return:
    '''
    print(path)
    with open(path, 'r') as fp:
        graph = json.load(fp)
    ori_nodes = graph['nodes']
    ori_edges = graph['edges']
    nodes_id = []
    max_x, max_y = -1, -1
    count = 0
    ori_no_5_nodes = []
    for node in ori_nodes:
        if node['type'] == 5:
            continue
        else:
            ori_no_5_nodes.append(node)
    graph_edges = []
    ori_node_id = []
    G = nx.DiGraph()  # 有向图保证顺序一致
    for node in ori_nodes:
        G.add_node(node['id'])
        ori_node_id.append(node['id'])
    for edge in ori_edges:
        graph_edges.append((edge['from'], edge['to']))
        # if edge['from'] not in ori_node_id:
        #     print('there is new node not in ori_nodes...','\n', edge['from'])
    G.add_edges_from(graph_edges)
    # print(G.nodes, '\n', ori_node_id)
    # print(G.edges, '\n', graph_edges)
    # print('init G nodes length: ', len(G.nodes))
    # print('init  ori_nodes length: ', len(ori_nodes))
    # nx.draw_networkx(G, with_labels=False)
    # plt.show()
    # 删除中间节点 type == 5 然后等价补全
    type5_num = 0
    for node in ori_nodes:
        if node['type'] == 5:
            type5_num += 1
            # 将原始数据中的对应边
            # for edge in ori_edges:
            #     if edge['from'] == node['id'] or edge['to'] == node['id']:
            #         ori_edges.remove(edge)
            aim_edge = filter(lambda edge : edge[1] == node['id'], G.edges).__next__()
            if aim_edge[1] == node['id']:

                from_node = aim_edge[0]
            for edge in G.edges:
                if edge[0] == node['id']:
                    to_node = edge[1]
                    G.add_edge(from_node, to_node)
                    #在原始数据中添加对应的边
                    ori_edges.append({'from' : from_node, 'to':to_node})
            G.remove_node(node['id'])
            # 删除原始数据中拐点
            # ori_nodes.remove(node)
    draw_graph_with_pos(G, ori_no_5_nodes, 'tp50')

    merge_graph = {'nodes': ori_nodes, 'edges': ori_edges}
    json_name = os.path.splitext(path)[-2].split('\\')[-1]
    # print('after merge G nodes length: ', len(G.nodes))
    # print('after merge  ori_nodes length: ', len(ori_nodes))
    if len(ori_no_5_nodes) != len(G.nodes):
        print('merge is wrong...')
    # SAVE_JSON(merge_graph, os.path.join('train_data/50_no_turnpts', json_name + '.json'))
    pos = []
    label = []
    for node in ori_no_5_nodes:
        pos.append(node['pos'])
        label.append(node['type'])
    # SAVE_JSON(merge_graph, os.path.join('train_dat
    # a/50_no_turnpts', os.path.splitext(path)[-2] + '.json') )
    # for node in nodes:
    #     x, y = node['pos']
    #     if x > max_x:
    #         max_x = x
    #     if y > max_y:
    #         max_y = y
    #     if node['type'] != 5:
    #         count += 1
    # print(count, f'\tmax_x:{max_x}', f'\tmax_y{max_y}')
    return G, np.array(pos), np.array(label)
def merge_all_graph():
    cwd = os.getcwd()
    NODES_NUM_TYPE_PATH = 'tp50'
    search_path = os.path.join(cwd, f'train_data\\{NODES_NUM_TYPE_PATH}')
    file_list = os.listdir(search_path)
    count = 0
    all_adj = []
    all_pos = []
    graph_order_info = {}
    order_index = 0
    for file in file_list:
        if os.path.splitext(file)[1] == '.json':
            G, single_pos, label = merge_one_graph(os.path.join(search_path, file))
            ad = nx.adjacency_matrix(G).todense()
            label = np.reshape(label, (-1, 1))

            adj = np.concatenate([ad, label], axis=1)
            print(adj.shape, label.shape)
            # all_adj.append(nx.adjacency_matrix(G).todense())
            all_adj.append(adj)
            all_pos.append(single_pos)
            count += 1
            graph ={'file': file, 'nodes': len(label)}
            graph_order_info[order_index] = graph
            order_index += 1
            print(count)
    # with open(f'train_data/{NODES_NUM_TYPE_PATH}_graph_order_info.json', 'w') as fp:
    #     json.dump(graph_order_info, fp)
    # np.save(f'train_data/{NODES_NUM_TYPE_PATH}_nodes_{count}_adj.npy', np.array(all_adj))
    # np.save(f'train_data/{NODES_NUM_TYPE_PATH}_nodes_{count}_pos.npy', np.array(all_pos))
def check_json_data_match_png():
    cwd = os.getcwd()
    NODES_NUM_TYPE_PATH = 'tp50'
    search_path = os.path.join(cwd, f'train_data\\{NODES_NUM_TYPE_PATH}')
    file_list = os.listdir(search_path)
    for file in file_list:
        if os.path.splitext(file)[1] == '.json':
            with open(os.path.join(search_path, file), 'r') as fp:
                graph = json.load(fp)
            print(file)
            ori_nodes = graph['nodes']
            ori_edges = graph['edges']
            graph_edges = []
            ori_node_id = []
            G = nx.DiGraph()
            nx_pos = {}
            for node in ori_nodes:
                G.add_node(node['id'])
                ori_node_id.append(node['id'])
                nx_pos[node['id']] = node['pos']
            for edge in ori_edges:
                graph_edges.append((edge['from'], edge['to']))
            G.add_edges_from(graph_edges)
            G = G.to_undirected()
            nx.draw_networkx(G, pos=nx_pos, with_labels=False, node_size = 5)
            plt.show()
def try_new():
    edges = [{'from': 1, 'to': 2}, {'from': 3, 'to': 4}]
    from_n = list(map(lambda e:e['from'] if e['to'] == 4 else None, edges))
    from_n = (filter(lambda edge : edge['to'] == 4, edges))
    print(from_n.__next__())
if __name__ == '__main__':
    # try_new()

    merge_all_graph()
    check_json_data_match_png()

