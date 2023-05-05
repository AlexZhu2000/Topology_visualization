import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import  PIL
RED_NODE_SIZE = 100
BLUE_NODE_SIZE = 100
YELLOW_NODE_SIZE = 100
BLACK_NODE_SIZE = 100
GREEN_NODE_SIZE = 20
BUSBAR_NODE_SIZE = 200

# icons = {
#         'root': r'F:\zzh\pycharm_program\TOPO_visualazition\icons\biandianzhan.png',
#         'root_switch': r'F:\zzh\pycharm_program\TOPO_visualazition\icons\root_switch.png',
#         'middle': r'F:\zzh\pycharm_program\TOPO_visualazition\icons\middle.png',
#         'contact_switch': r'F:\zzh\pycharm_program\TOPO_visualazition\icons\contact_switch.png'
#     }
#
# node_icons = {k : PIL.Image.open(fname) for k, fname in icons.items()}

class My_Graph:
    '''
    node : 节点
    edge : (a, b)
    pos :[[x, y], [x, y], ... , []]
    '''
    count = 0
    def __init__(self, node, root, busbar, root_switch, middle, contact_switch, edge, dict_pos):
        self.node = node
        len_node = len(node)
        self.edge = edge
        self.root = root
        self.busbar = busbar
        self.root_switch = root_switch
        self.middle = middle
        self.contact_switch = contact_switch
        self.pos = dict_pos
        My_Graph.count += 1
        self.G = nx.Graph()
        self.G.add_nodes_from(node)
        self.G.add_edges_from(edge)
        self.adjacency_matrix = np.array(nx.adjacency_matrix(self.G).todense())
        color_map = []
        for i in node:
            if i in root:
                color_map.append('red')
            elif i in busbar:
                color_map.append('yellow')
            elif i in root_switch:
                color_map.append('black')
            elif i in middle:
                color_map.append('green')
            elif i in contact_switch:
                color_map.append('blue')
        self.color_map = color_map
    def record_to_file(self):
        outfile = open(f'graph_data\\{My_Graph.count}.pkl', 'wb')
        OK = pickle.dump(self, outfile)
        outfile.close()
        return OK
    def load_from_file(self):
        pkl_file = open(f'graph_data\\{My_Graph.count}.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data
    def get_adjacency_matrix(self):
        adjacency_matrix = np.array(nx.adjacency_matrix(self.G).todense())
        return adjacency_matrix
    def get_label(self):
        label = np.zeros_like(self.node)
        label[self.root] = 1
        label[self.root_switch] = 2
        label[self.middle] = 3
        label[self.contact_switch] = 4
    def show_graph(self):
        node_size = np.zeros_like(self.node)
        index_red = np.where(np.array(self.color_map) == 'red')
        index_blue = np.where(np.array(self.color_map) == 'blue')
        index_green = np.where(np.array(self.color_map) == 'green')
        index_yellow = np.where(np.array(self.color_map) == 'yellow')
        index_black = np.where(np.array(self.color_map) == 'black')
        node_size[index_red] = RED_NODE_SIZE
        node_size[index_blue] = BLUE_NODE_SIZE
        node_size[index_yellow] = YELLOW_NODE_SIZE
        node_size[index_black] = BLACK_NODE_SIZE
        node_size[index_green] = GREEN_NODE_SIZE
        # self.G.remove_edges_from(self.G.edges)
        # print(f'the node_size list is {node_size}...')
        nx.draw_networkx(self.G, pos=self.pos, node_color=self.color_map, node_size=node_size, with_labels=False)
        #　plt.show()　＃训练GAN过程中不需要此plt.show(), 其他过程如果要显示则此处需要plt.show()

        # fig, ax = plt.subplots()
        #
        # nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=self.G.edges, ax=ax, arrows=True)
        # tr_figure = ax.transData.transform
        # tr_axes = fig.transFigure.inverted().transform
        # nx.draw_networkx_nodes(self.G, self.pos, nodelist=np.arange(30), node_size=np.zeros((30)))
        # icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
        # icon_center = icon_size / 2.0
        # middle_size = icon_size / 3.
        # middle_center = middle_size / 2.
        # for n in self.G.nodes:
        #     xf, yf = tr_figure(self.pos[n])
        #     # print(xf, yf)
        #     xa, ya = tr_axes((xf, yf))
        #     # print('----', xa, ya)
        #
        #     if n in self.root:
        #         a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        #         a.imshow(node_icons['root'])
        #     elif n in self.root_switch:
        #         a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        #         a.imshow(node_icons['root_switch'])
        #     elif n in self.middle:
        #         a = plt.axes([xa - middle_center, ya - middle_center, middle_size, middle_size])
        #         a.imshow(node_icons['middle'])
        #     elif n in self.contact_switch:
        #         a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
        #         a.imshow(node_icons['contact_switch'])
        #     a.axis('off')
        return True