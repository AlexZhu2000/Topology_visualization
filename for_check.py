import networkx as nx
import numpy as np
import json





class Machine():
    def __init__(self, node_size, pos, adj, adj_num):
        self.nodes = np.arange(node_size)
        self.pos = pos
        self.adj_num = adj_num
        self.ad = adj[:, :-1]
        self.label = adj[:, -1]
        self.G = nx.from_numpy_matrix(self.ad)
    def switch_to_json(self):

        # print(self.G.edges)
        nodes = []
        for node in self.nodes:
            node_dict = {
                'id':int(node),
                'pos':list(self.pos[node]),
                'label':str(int(self.label[node]))
            }

            nodes.append(node_dict)
        print(nodes)
        edges = []
        for edge in self.G.edges:
            a, b = edge[0], edge[1]
            edges_josn = {
                'from_node':a,
                'to_node' : b
            }
            edges.append(edges_josn)
        print(edges)
        all_json = {
            'nodes':nodes,
            'edges':edges
        }
        with open(f'for_check\\json\\{self.adj_num}.json', "w") as f:
            json.dump(all_json, f)

        return 1

if __name__ == '__main__':
    adj = np.load('train_data\\3000_adj.npy').astype('float64')
    train_pos = np.load('train_data\\3000_pos.npy').astype('float64')
    adj_num = 0
    for adj_num in range(10):
        m = Machine(30, train_pos[adj_num], adj[adj_num], adj_num=adj_num)
        m.switch_to_json()