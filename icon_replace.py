import networkx as nx
import PIL
import matplotlib.pyplot as plt
import numpy as np
from My_Graph_Class import My_Graph
import nodes_connections_layout as Nodes_Connection
def icons_try():
    icons = {
        'root': r'icons\biandianzhan.png',
        'root_switch': r'icons\root_switch.png',
        'middle': r'icons\middle.png',
        'contact_switch': r'icons\contact_switch.png'
    }

    node_icons = {k : PIL.Image.open(fname) for k, fname in icons.items()}
    for key in node_icons.keys():
        plt.figure()
        plt.imshow(node_icons[key])
        plt.show()
icons = {
        'root': r'icons\biandianzhan.png',
        'root_switch': r'icons\root_switch.png',
        'middle': r'icons\middle.png',
        'contact_switch': r'icons\contact_switch.png'
    }

node_icons = {k : PIL.Image.open(fname) for k, fname in icons.items()}
# for key in node_icons.keys():
#     plt.figure()
#     plt.imshow(node_icons[key])
#     plt.show()

def draw_topo_from_pos(pos, adj):
    '''

    :param pos:
    :param ad:
    :return:
    '''
    nodes_label = adj[:, -1]
    ad = adj[:, :-1]
    root = np.argwhere(nodes_label == 1).flatten()
    root_switch = np.argwhere(nodes_label == 2).flatten()
    middle = np.argwhere(nodes_label == 3).flatten()
    contact_switch = np.argwhere(nodes_label == 4).flatten()

    dict_pos = dict.fromkeys(np.arange(30))
    for i in range(len(pos)):
        dict_pos[i] = tuple(pos[i] / 8.)

    graph = nx.from_numpy_matrix(ad)
    # fig, ax = plt.subplots()
    #
    # nx.draw_networkx_edges(graph, pos=dict_pos, edgelist=graph.edges, ax=ax, arrows=True)
    # tr_figure = ax.transData.transform
    # tr_axes = fig.transFigure.inverted().transform
    # nx.draw_networkx_nodes(graph, dict_pos, nodelist=np.arange(30), node_size=np.zeros((30)))
    # icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
    # icon_center = icon_size / 2.0
    # for n in graph.nodes:
    #     xf, yf = tr_figure(dict_pos[n])
    #     # print(xf, yf)
    #     xa, ya = tr_axes((xf, yf))
    #     # print('----', xa, ya)
    #     a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
    #     a.imshow(node_icons['root'])
    #     a.axis('off')
    # # print(ax.get_xlim()[1] - ax.get_xlim()[0])
    # #plt.axis('off')
    # plt.show()
    my_graph = My_Graph(np.arange(30), root, root_switch, middle, contact_switch, graph.edges, dict_pos)
    Nodes_Connection.draw_topo_from_graph(my_graph)
    plt.show()
if __name__ == '__main__':
    adj = np.load('train_data\\3000_adj.npy')
    pos = np.load('train_data\\3000_pos.npy')
    draw_topo_from_pos(pos[0], adj[0])