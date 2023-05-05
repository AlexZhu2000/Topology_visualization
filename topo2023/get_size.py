import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import shutil
NODES_50_ADJ_NPY_PATH = 'train_data/tp50_nodes_818_adj.npy'
NODES_50_POS_NPY_PATH = 'train_data/tp50_nodes_818_pos.npy'
NODES_100_ADJ_NPY_PATH = 'train_data/tp100_nodes_719_adj.npy'
NODES_100_POS_NPY_PATH = 'train_data/tp100_nodes_719_pos.npy'
NODES_50_POS_NORMAL_NPY_PATH = 'train_data/tp50_nodes_818_graph_normalize_pos.npy'
def xy_normalization(all_pos):
    '''
    all_pos : (n, node_num, 2)numpy
    将坐标的绝对值转换为标准化值，以获得棋盘大小
    :return:
    '''
    all_new_normal_pos = []
    all_graph_x_len = []
    all_graph_y_len = []
    all_nodes_len = []
    for graph_pos in all_pos:
        all_nodes_len.append(len(graph_pos))
        graph_x_list = []
        graph_y_list = []
        for single_pos in graph_pos:
            if single_pos[0] not in graph_x_list:
                graph_x_list.append(single_pos[0])
            if single_pos[1] not in graph_y_list:
                graph_y_list.append(single_pos[1])
        graph_x_len = len(graph_x_list)
        graph_y_len = len(graph_y_list)
        all_graph_x_len.append(graph_x_len)
        all_graph_y_len.append(graph_y_len)
        '''
        将x y值等序替换成归一化后的正整数值0, 1, 2, 3....
        '''
        graph_x_list.sort()
        graph_y_list.sort()
        graph_x_pair_dict = {}
        graph_y_pair_dict = {}
        # 得到x\y值等效替换为正整数的字典
        for index, x_value in enumerate(graph_x_list):
            graph_x_pair_dict[x_value] = index
        for index, y_value in enumerate(graph_y_list):
            graph_y_pair_dict[y_value] = index
        new_graph_pos = []
        #利用字典将坐标转换为正整数
        for single_pos in graph_pos:
            new_x = graph_x_pair_dict[single_pos[0]]
            new_y = graph_y_pair_dict[single_pos[1]]
            new_graph_pos.append((new_x, new_y))
        #合并到新的总pos
        all_new_normal_pos.append(new_graph_pos)
    all_graph_new_normal_pos = np.array(all_new_normal_pos)
    np.save('train_data/tp100_nodes_719_graph_normalize_pos.npy', all_graph_new_normal_pos)
    print('min_x : ', min(all_graph_x_len), 'max_x : ', max(all_graph_x_len))
    print('min_y : ', min(all_graph_y_len), 'max_y : ', max(all_graph_y_len))
    print('max_nodes_num:', max(all_nodes_len), 'min_nodes_num', min(all_nodes_len))
    '''
    得到最大的x y 值之后，将
    '''
def show_graph_to_check_merge():
    '''
    使用原始x,y值进行可视化呈现，坐标值如[660, 360]
    :return:
    '''
    all_adj = np.load(NODES_50_ADJ_NPY_PATH, allow_pickle=True)
    all_pos = np.load(NODES_50_POS_NPY_PATH, allow_pickle=True)
    for adj, pos in zip(all_adj, all_pos):
        ad = adj[:, :-1]
        nx_pos = {}
        print(adj.shape, len(pos))
        for i in range(len(pos)):
            nx_pos[i] = pos[i]
        G = nx.from_numpy_matrix(ad)
        nx.draw_networkx(G, pos = nx_pos, node_size=5, with_labels=False)
        plt.show()
def show_normal_graph_to_check_merge():
    '''
    将坐标x y按照原序列替换成正整数序列0, 1 ,2.....后，坐标值如[2, 5]
    :return:
    '''
    all_adj = np.load(NODES_50_ADJ_NPY_PATH, allow_pickle=True)
    all_pos = np.load(NODES_50_POS_NORMAL_NPY_PATH, allow_pickle=True)
    for adj, pos in zip(all_adj, all_pos):
        ad = adj[:, :-1]
        nx_pos = {}
        print(adj.shape, len(pos))
        for i in range(len(pos)):
            nx_pos[i] = pos[i]
        G = nx.from_numpy_matrix(ad)
        nx.draw_networkx(G, pos = nx_pos, node_size=5, with_labels=False)
        plt.show()
def transfer_png_file():
    old_path = 'train_data/history/tp50'
    dst_path = 'train_data/tp50'
    file_list = os.listdir(old_path)
    count = 0
    p_count = 0
    for file in file_list:
        if os.path.splitext(file)[1] == '.json':
            count += 1
        if os.path.splitext(file)[1] == '.png':
            p_count += 1
            # shutil.copyfile(os.path.join(old_path, file), os.path.join(dst_path, file))
    print(count, p_count)
if __name__ == '__main__':
    all_pos = np.load(NODES_100_POS_NPY_PATH, allow_pickle=True)
    show_normal_graph_to_check_merge()
    xy_normalization(all_pos)
