import numpy as np
import networkx as nx
import json
class Layout_xy():
    def __init__(self, pos, adj, nodesize):
        self.pos = pos
        self.adj = adj
        self.nodesize = nodesize
    def output_json(self):
        nodes = []
        edges = []
        all_label = self.adj[:, -1]
        for index, node in enumerate(self.adj):
            nodes.append({'id': index, 'pos': list(self.pos[index]), 'label': int(all_label[index])})
            print(self.adj.shape, index, all_label[index])
            g = nx.from_numpy_matrix(self.adj[:, :-1])
            for edge in g.edges():
                edges.append({'from': edge[0], 'to': edge[1]})
        dict = {'nodes': nodes, 'edges': edges}
        with open(f'for_tomax\\0403\\{self.nodesize}_{np.random.random()}.json', "w") as f:
            json.dump(dict, f)
def transfer(num):
    all_pos = np.load('train_data/tp50_nodes_818_graph_normalize_pos.npy', allow_pickle=True)
    all_adj = np.load('train_data/tp50_nodes_818_adj.npy', allow_pickle=True)


    # for (adj, pos) in zip(all_adj[:num], all_pos[:num]):
    #     nodes = []
    #     edges = []
    #     print(pos)
    #     for index, node in enumerate(adj):
    #         label = node[-1]
    #         nodes.append({'id' : index, 'pos': list(pos[index]), 'label' : label})
    #         print(index, pos.shape, label)
    #         g = nx.from_numpy_matrix(adj[: , :-1])
    #         for edge in g.edges():
    #             edges.append({'from':edge[0], 'to':edge[1]})
    #
    #     dict = {'nodes': nodes, 'edges':edges}
    #     with open(f'for_tomax\\{num}.json', "w") as f:
    #         json.dump(dict, f)
    for i in range(num):
        nodes = []
        edges = []
        single_adj = all_adj[i]
        single_pos = all_pos[i]
        all_label = single_adj[:, -1]
        for index, node in enumerate(single_adj):
            nodes.append({'id' : index, 'pos': list(single_pos[index]), 'label' : int(all_label[index])})
            print(single_adj.shape, index, all_label[index])
            g = nx.from_numpy_matrix(single_adj[:, :-1])
            for edge in g.edges():
                edges.append({'from': edge[0], 'to': edge[1]})
        dict = {'nodes': nodes, 'edges': edges}
        with open(f'for_tomax\\{i+1}.json', "w") as f:
            json.dump(dict, f)
def output_single_json(single_adj, single_pos, score):
    nodes = []
    edges = []
    all_label = single_adj[:, -1]
    for index, node in enumerate(single_adj):
        nodes.append({'id': index, 'pos': list(single_pos[index]), 'label': int(all_label[index])})
        print(single_adj.shape, index, all_label[index])
        g = nx.from_numpy_matrix(single_adj[:, :-1])
        for edge in g.edges():
            edges.append({'from': edge[0], 'to': edge[1]})
    dict = {'nodes': nodes, 'edges': edges}
    with open(f'for_tomax\\{score}.json', "w") as f:
        json.dump(dict, f)
def chosse_nodesize_to_transfer(nodesize):
    '''
    根据0403需求：选择40、50节点规模的训练数据输出，选择（x, y ）形式坐标而不是棋盘坐标
    :param nodesize:40 or 50
    :return:
    '''
    all_pos = np.load('train_data/tp50_nodes_818_graph_normalize_pos.npy', allow_pickle=True)
    all_adj = np.load('train_data/tp50_nodes_818_adj.npy', allow_pickle=True)
    for (pos, adj) in zip(all_pos, all_adj):
        if len(adj) == nodesize:
            layout = Layout_xy(pos, adj, nodesize)
            layout.output_json()

if __name__ == '__main__':
    # transfer(5)
    # all_pos = np.load('train_data/tp100_chessboard_pos.npy', allow_pickle=True)
    # all_adj = np.load('train_data/tp100_nodes_719_adj_padding.npy', allow_pickle=True)
    # print(all_pos.shape, all_adj.shape)
    chosse_nodesize_to_transfer(50)
