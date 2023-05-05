import numpy as np
import copy
from My_Graph_Class import My_Graph
def if_two_pos_orthogonal(pos1, pos2):
    '''
    find out if these two positiions are orthogonal
    :param pos1:
    :param pos2:
    :return:0 or 1
    '''
    (a, b), (x, y) = pos1, pos2
    if a == x or b == y:
        return 1
    else:
        return 0
def relationship_between_root_and_leaf(root, leaf):
    '''
    判断两个节点之间的位置关系，来决定连接的路线安排；
    root作为二次坐标原点，位置关系分为五个：四个象限（1, 2, 3, 4）以及坐标轴（0）
    :param root:
    :param leaf:
    :return:
    '''
    (a, b), (x, y) = root, leaf
    if if_two_pos_orthogonal(root, leaf):
        return 0
    elif x > a and y > b:
        return 1
    elif x < a and y > b:
        return 2
    elif x < a and y < b:
        return 3
    elif x > a and y < b:
        return 4
def whether_straight_connection_conflict(node1, pos1, node2, pos2, new_graph):
    '''
    已知两点正交，pos1和pos2直接连线，判断是否和new_graph中节点重叠或者线重叠
    :return:0 or 1
    '''
    (x1, y1), (x2, y2) = list(pos1), list(pos2)
    if x1 == x2:
        # 如果在同一直线上
        # 判断连线是否经过某个节点
        for node_num, pos in new_graph.pos.items():
            if node_num == node1 or node_num == node2:
                continue
            (temp_x, temp_y) = pos
            # if pos == pos1 or pos == pos2:
            #     continue
            if temp_x == x1:
                xlabel2 = (temp_y - y1) * (temp_y - y2)
                if xlabel2 < 0:
                    return True
        for edge in new_graph.edge:
            # 判断边是否有重叠
            if (node1, node2) == edge or (node2, node1) == edge:
                continue
            a, b = edge[0], edge[1]
            (ax, ay), (bx, by) = new_graph.pos[a], new_graph.pos[b]
            if ax == bx == x1:
                max_len = max(ay, by, y1, y2) - min(ay, by, y1, y2)
                total_len = abs(y1 - y2) + abs(ay - by)
                if total_len > max_len:
                    return True
        return False
    elif y1 == y2:
        for node_num, pos in new_graph.pos.items():
            if node_num == node1 or node_num == node2:
                continue
            (temp_x, temp_y) = pos
            # if pos == pos1 or pos == pos2:
            #     continue
            if temp_y == y1:
                ylabel2 = (temp_x - x1) * (temp_x - x2)
                if ylabel2 < 0:
                    return  True
        for edge in new_graph.edge:
            # 判断边是否有重叠
            if (node1, node2) == edge or (node2, node1) == edge:
                continue
            a, b = edge[0], edge[1]
            (ax, ay), (bx, by) = new_graph.pos[a], new_graph.pos[b]
            if ay == by == y1:
                max_len = max(ax, bx, x1, x2) - min(ax, bx, x1, x2)
                total_len = abs(x1 - x2) + abs(ax - bx)
                if total_len > max_len:
                    return True
        return False
def whether_turn_overlapping(turn, new_graph):
    '''
    判断拐点是否和目前图的节点重叠
    :param turn:
    :param new_graph:
    :return:
    '''
    for node_num, pos in new_graph.pos.items():
        if pos == turn:
            return True
    return False
def whether_OneTurnPoint_connection_conflict(node1, pos1, node2, pos2, new_graph):
    '''
    用一个拐点的方式连接，在new_graph中查看是否根节点重叠
    可以利用whether_straight_connection_conflict
    :return:TurnPoint or 1
    '''
    (x1, y1), (x2, y2) = pos1, pos2
    turn_point1 = (x1, y2)
    turn_point2 = (x2, y1)
    for node_num, pos in new_graph.pos.items():
        if pos == turn_point2 or pos == turn_point1:
            '''
            如果拐点与先前的点重合，则直接返回冲突
            '''
            return True
    if whether_straight_connection_conflict(node1, pos1, -1,turn_point1, new_graph) or\
        whether_straight_connection_conflict(-1, turn_point1, node2, pos2, new_graph):
        # 如果添加的一种拐点方法导致冲突，判断第二种拐点方法是否冲突
        if whether_straight_connection_conflict(node1, pos1,-1,  turn_point2, new_graph) or\
            whether_straight_connection_conflict(-1, turn_point2, node2, pos2, new_graph):
            # 如果都冲突，说明一个拐点导致冲突返回1
            return True
        else:
            return turn_point2
    else:
        return turn_point1
def drawboard_analyze(new_graph):
    '''
    为了合理安排连线的拐点位置，需要了解整个画板的性质，再决定拐点的安排
    :param new_graph:
    :return:width, height
    '''
    min_x, min_y = 99., 99.
    max_x, max_y = -1., -1.
    for node_num, pos in new_graph.pos.items():
        (x, y) = pos
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
    return (max_x - min_x), (max_y - min_y)
def Orthogonal_TwoTurnPoint_connection(node1, pos1, node2, pos2, new_graph):
    '''
    对于正交的两个节点，添加两个拐点进行连线；
    先考虑连线统一安排在两点的右下方；
    根据经验，需要两个拐点连线的正交的两个叶子，更有可能是竖向排列
    1:
    node1 ------- turn1
                    |
                    |
                    |
                    |
                    |
    node2 ------ turn2
    2:
    turn1 ----------------- turn2
        |                       |
        |                       |
        |                       |
    node1                   node2

    :param pos1:
    :param pos2:
    :param new_graph:
    :return:
    '''
    (x1, y1), (x2, y2) = pos1, pos2
    drawboard_width, drawboard_height = drawboard_analyze(new_graph)

    x_interval = drawboard_width / 12.
    y_interval = drawboard_height / 10.
    if x1 == x2:
        if x1 == drawboard_width / 2:
            interval_dir = -1
        else:
            interval_dir = (drawboard_width / 2 - x1) / abs(x1 - drawboard_width / 2)
        for i in range(3):
            # 如果是在同一竖线上(drawboard_width / 2 - x1)/abs(x1 - drawboard_width / 2)为前进方向 [-1, 1]
            turn1 = (x1 + (i + 1) * interval_dir * x_interval, y1)
            turn2 = (x1 + (i + 1) * interval_dir * x_interval, y2)
            if whether_turn_overlapping(turn2, new_graph) or\
                whether_turn_overlapping(turn1, new_graph) or\
                whether_straight_connection_conflict(node1, pos1, -1, turn1, new_graph) or\
                whether_straight_connection_conflict(-1, turn1, -1, turn2, new_graph) or\
                whether_straight_connection_conflict(-1,turn2,node2, pos2, new_graph):
                continue
            new_graph = update_graph(node1, pos1, node2, pos2,new_graph, turn1, turn2)
            return new_graph
    elif y1 == y2:
        if y1 == drawboard_height / 2:
            interval_dir = -1
        else:
            interval_dir = (drawboard_height / 2 - y1) / abs(y1 - drawboard_height / 2)
        for i in range(3):
            turn1 = (x1,  y1 + (i + 1) * interval_dir * y_interval)
            turn2 = (x2,  y2 + (i + 1) * interval_dir * y_interval)
            if whether_turn_overlapping(turn2, new_graph) or\
                whether_turn_overlapping(turn1, new_graph) or\
                whether_straight_connection_conflict(node1, pos1,-1, turn1, new_graph) or\
                whether_straight_connection_conflict(-1, turn1,-1, turn2, new_graph) or\
                whether_straight_connection_conflict(-1, turn2,node2, pos2, new_graph):
                continue
            new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
            return new_graph
    return new_graph
def Not_Orthogonal_TwoTurnPoint_connection(node1, pos1, node2, pos2, new_graph):
    '''
    1 中间节点：
    node1 ------ turn1
                    |
                    |
                    |
                    |
                 turn2 --------- node2
    2：
    node1
       |
       |
       |
       |
    turn1 ------ turn2
                    |
                    |
                    |
                    |
                  node2
    对于不正交的两个节点，添加两个拐点进行连接
    考虑直接从高点开始，在水平距离的中间出添加一个拐点，在垂直距
    3:
    turn1------------------turn2
    |                       |
    |                       |
    node1                   |
                            |
                            |
                            |
                            node2
    4:
    node1---------------------------------turn1
                                            |
                                            |
                                            |
                                            |
                                            |
                                            |
                        node2--------------turn2
    :param pos1:
    :param pos2:
    :param new_graph:
    :return:
    '''
    width, height = drawboard_analyze(new_graph)
    (x1, y1), (x2, y2) = pos1, pos2
    # y_interval = (y2 - y1) / 5.
    y_interval = height / 15.
    # 尝试第2种
    for i in range(4):
        turn1 = (x1, y1 + y_interval * (i + 1))
        turn2 = (x2, y1 + y_interval * (i + 1))
        if whether_straight_connection_conflict(node1, pos1, -1, turn1, new_graph) or \
                whether_straight_connection_conflict(-1, turn1,-1, turn2, new_graph) or \
                whether_straight_connection_conflict(-1, turn2,node2, pos2, new_graph):
            continue
        new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
        return new_graph
    # 尝试第1种
    # x_interval = (x2 - x1) / 5.  # 带方向的间隔，不加绝对值
    x_interval = width / 15.
    for i in range(4):
        # 两种不同间隔大小的尝试
        turn1 = (x1 + x_interval * (i + 1), y1)
        turn2 = (x1 + x_interval * (i + 1), y2)
        if whether_turn_overlapping(turn2, new_graph) or \
                whether_turn_overlapping(turn1, new_graph) or \
                whether_straight_connection_conflict(node1, pos1,-1, turn1, new_graph) or \
                whether_straight_connection_conflict(-1, turn1,-1, turn2, new_graph) or \
                whether_straight_connection_conflict(-1, turn2,node2, pos2, new_graph):
            continue
        new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
        return new_graph

    #尝试第3种向上
    for i in range(4):
        turn1 = (x1, max(y1, y2) + abs(y_interval) * (i + 1))
        turn2 = (x2, max(y1, y2) + abs(y_interval) * (i + 1))
        if whether_turn_overlapping(turn2, new_graph) or \
                whether_turn_overlapping(turn1, new_graph) or \
                whether_straight_connection_conflict(node1, pos1,-1, turn1, new_graph) or \
                whether_straight_connection_conflict(-1, turn1,-1, turn2, new_graph) or \
                whether_straight_connection_conflict(-1, turn2,node2, pos2, new_graph):
            continue
        new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
        return new_graph
    # 尝试第3种向下
    for i in range(4):
        turn1 = (x1, min(y1, y2) - abs(y_interval) * (i + 1))
        turn2 = (x2, min(y1, y2) - abs(y_interval) * (i + 1))
        if whether_turn_overlapping(turn2, new_graph) or \
                whether_turn_overlapping(turn1, new_graph) or \
                whether_straight_connection_conflict(node1, pos1,-1, turn1, new_graph) or \
                whether_straight_connection_conflict(-1, turn1,-1, turn2, new_graph) or \
                whether_straight_connection_conflict(-1, turn2,node2, pos2, new_graph):
            continue
        new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
        return new_graph
    # 尝试第4种向右
    for i in range(4):
        turn1 = (max(x1, x2) + abs(x_interval) * (i + 1), y1)
        turn2 = (max(x1, x2) + abs(x_interval) * (i + 1), y2)
        if whether_turn_overlapping(turn2, new_graph) or \
                whether_turn_overlapping(turn1, new_graph) or \
                whether_straight_connection_conflict(node1, pos1,-1, turn1, new_graph) or \
                whether_straight_connection_conflict(-1, turn1, -1, turn2, new_graph) or \
                whether_straight_connection_conflict(-1, turn2, node2, pos2, new_graph):
            continue
        new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
        return new_graph
    # 尝试第4种向左
    for i in range(4):
        turn1 = (min(x1, x2) - abs(x_interval) * (i + 1), y1)
        turn2 = (min(x1, x2) - abs(x_interval) * (i + 1), y2)
        if whether_turn_overlapping(turn2, new_graph) or \
                whether_turn_overlapping(turn1, new_graph) or \
                whether_straight_connection_conflict(node1, pos1,-1, turn1, new_graph) or \
                whether_straight_connection_conflict(-1, turn1,-1, turn2, new_graph) or \
                whether_straight_connection_conflict(-1, turn2,node2, pos2, new_graph):
            continue
        new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
        return new_graph
    # 保底
    turn1 = (x1, y1 + y_interval * (i + 1))
    turn2 = (x2, y1 + y_interval * (i + 1))

    new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
    return new_graph

    # if abs(y1 - y2) > abs(x1 - x2):
    #     #第二种，竖向距离较长
    #     interval = (y2 - y1) / 5.
    #     for i in range(4):
    #         # 两种不同间隔大小的尝试
    #         turn1 = (x1, y1 + interval * (i + 1))
    #         turn2 = (x2, y1 + interval * (i + 1))
    #         if whether_straight_connection_conflict(pos1, turn1, new_graph) or \
    #                 whether_straight_connection_conflict(turn1, turn2, new_graph) or \
    #                 whether_straight_connection_conflict(turn2, pos2, new_graph):
    #             # 如果方法2无效，则选择方法4
    #             turn1 = (max(x1, x2))
    #             continue
    #         new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
    #         return new_graph
    #     new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
    #     return new_graph
    # else:
    #     # 第一种，横向距离较长
    #     interval = (x2 - x1) / 5.  # 带方向的间隔，不加绝对值
    #     for i in range(4):
    #         # 两种不同间隔大小的尝试
    #         turn1 = (x1 + interval * (i + 1), y1)
    #         turn2 = (x1 + interval * (i + 1), y2)
    #         if whether_turn_overlapping(turn2, new_graph) or\
    #             whether_turn_overlapping(turn1, new_graph) or\
    #             whether_straight_connection_conflict(pos1, turn1, new_graph) or\
    #             whether_straight_connection_conflict(turn1, turn2, new_graph) or\
    #             whether_straight_connection_conflict(turn2, pos2, new_graph):
    #             continue
    #         new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
    #         return new_graph
    #     new_graph = update_graph(node1, pos1, node2, pos2, new_graph, turn1, turn2)
    #     return new_graph
def update_graph(node1, pos1, node2, pos2, new_graph, new_point1 = None, new_point2 = None):
    '''
    node1 (------- new_point1 (-------new_point2)) ------ node2
    已知在某两个节点中添加节点（1 or 2），
    现在更新图结构中的连线、节点数、节点颜色等数据以绘制合理拓扑图
    :param node1: 被改变连线的一个节点
    :param pos1: node1的位置(x, y)
    :param node2: 被改变连线的另一个节点
    :param pos2: node2的位置(x, y)
    :param new_graph: 当前的图数据结构
    :param new_point1: 添加的新拐点坐标（x， y）
    :param new_point2: 添加的第二个拐点坐标(x, y)
    :return: 更新后的graph数据结构
    '''
    if new_point1 == None:
        #如果没有新添加的拐点，则直接添加边
        new_graph.G.add_edge(node1, node2)
        return new_graph
    elif new_point2 == None:
        #如果只添加了一个节点
        node = len(new_graph.node)
        new_graph.node = np.append(new_graph.node, node)  # 更新Node 防止添加坐标时出错
        new_graph.G.add_node(node)
        new_graph.G.add_edges_from([(node1, node), (node, node2)])  # 添加拐点，并连接
        new_graph.G.remove_edge(node1, node2)  # 添加拐点后，删除原来的连线
        new_graph.color_map.append('yellow')
        new_graph.pos[node] = new_point1
        return new_graph
    else:
        new_node1 = len(new_graph.node)
        new_node2 = new_node1 + 1
        new_graph.node = np.append(new_graph.node, [new_node1, new_node2])
        new_graph.G.add_node(new_node1)
        new_graph.G.add_node(new_node2)
        new_graph.color_map.append('yellow')
        new_graph.color_map.append('yellow')
        new_graph.G.add_edges_from([(node1, new_node1), (new_node1, new_node2), (new_node2, node2)])#添加边
        new_graph.G.remove_edge(node1, node2) #删除原来的边
        new_graph.pos[new_node1] = new_point1
        new_graph.pos[new_node2] = new_point2
        return new_graph
CELL_LENGTH = 0.125
ROOT_SWITCH_OFFSET = CELL_LENGTH / 4.
def change_node_pos(new_graph):
    width, height = drawboard_analyze(new_graph)
    # print('width:', width, 'height:', height)
    x_offset = width / 10.
    y_offset = height / 10.
    root_x_record = []
    root_switch_x_record = []
    middle_x_record = []
    contact_x_record = []
    for node in new_graph.node:
        if node in new_graph.root:
            (x, y) = new_graph.pos[node]
            while x in root_x_record:
                x -= x_offset / 5.
            # if x in root_x_record:
            #     x += x_offset / 3.
            root_x_record.append(x)
            new_graph.pos[node] = (x, y)
        elif node in new_graph.root_switch:
            '''
            先改变坐标位置，规避线重合的情况
            '''
            (x, y) = new_graph.pos[node]
            while x in root_switch_x_record:
                x -= x_offset / 4.
            # if x < width / 2.:
            #     y += y_offset / 3.
            root_switch_x_record.append(x)
            new_graph.pos[node] = (x, y)
        elif node in new_graph.middle:
            '''
            '''
            (x, y) = new_graph.pos[node]
            # if x < width / 2.:
            #     y += 2 * (y_offset / 12.)
            while x in middle_x_record:
                x += x_offset / 7.
            middle_x_record.append(x)
            new_graph.pos[node] = (x, y)
        elif node in new_graph.contact_switch:
            '''
            '''
            (x, y) = new_graph.pos[node]
            if x < width / 2.:
                y -= 2 * (y_offset / 4.)
            elif x > width / 2:
                y += 3 * (y_offset / 5.)
            else:
                y += (y_offset / 6.)
            while x in contact_x_record:
                x += x_offset / 7.
            contact_x_record.append(x)
            new_graph.pos[node] = (x, y)
    return new_graph
def draw_topo_from_graph(my_graph):
    '''
    层次连线算法
    :param my_graph:
    :return:
    '''

    new_graph = copy.deepcopy(my_graph)
    # new_graph = change_node_pos(new_graph)
    if new_graph == None:
        print('xxxxxxxxxxxxx')
    for root in my_graph.root:
        # 对于每个根节点
        my_leaves = np.argwhere(my_graph.adjacency_matrix[root] == 1).flatten()
        # 判断跟节点和叶子节点是否处在同一正交线路上
        for leaf in my_leaves:
            if if_two_pos_orthogonal(new_graph.pos[root], new_graph.pos[leaf]):
            #if relationship_between_root_and_leaf(my_graph.pos[root], my_graph.pos[leaf]) == 0:
                # 如果正交，那么直接相连（判断线路上是否有其他节点）
                None
            else:
                # 如果不正交，添加节点（判断线路上是否有其他节点）
                # 判断叶子节点和根的位置关系，决定连接线路
                root_pos, leaf_pos= (root_x, root_y), (leaf_x, leaf_y) = new_graph.pos[root], new_graph.pos[leaf]
                res = whether_OneTurnPoint_connection_conflict(root, root_pos, leaf, leaf_pos, my_graph)
                if res == 1:
                    new_graph = Not_Orthogonal_TwoTurnPoint_connection(root, root_pos, leaf, leaf_pos, new_graph)
                else:
                    update_graph(root, root_pos, leaf, leaf_pos, new_graph, res)
                # relative_pos = relationship_between_root_and_leaf(root_pos, leaf_pos)
                # if relative_pos == 1:
                #     # cross_pos = (root_x, leaf_y)
                #     # node = len(new_graph.node)
                #     # new_graph.node = np.append(new_graph.node, node) #更新Node 防止添加坐标时出错
                #     # new_graph.G.add_node(node)
                #     # new_graph.G.add_edges_from([(root, node), (node, leaf)]) #添加拐点，并连接
                #     # new_graph.G.remove_edge(root, leaf) #添加拐点后，删除原来的连线
                #     # new_graph.color_map.append('yellow')
                #     # new_graph.pos[node] = cross_pos
                #     res = whether_OneTurnPoint_connection_conflict(root_pos, leaf_pos, my_graph)
                #     if res == 1:
                #         new_graph = Not_Orthogonal_TwoTurnPoint_connection(root, root_pos, leaf, leaf_pos, new_graph)
                #     else:
                #         update_graph(root, root_pos, leaf, leaf_pos, new_graph, res)
                # elif relative_pos == 2:
                #     # cross_pos = (root_x, leaf_y)
                #     # node = len(new_graph.node)
                #     # new_graph.node = np.append(new_graph.node, node)
                #     # new_graph.G.add_node(node)
                #     # new_graph.G.add_edges_from([(root, node), (node, leaf)])
                #     # new_graph.G.remove_edge(root, leaf)
                #     # new_graph.color_map.append('yellow')
                #     # new_graph.pos[node] = cross_pos
                #     # None
                #     res = whether_OneTurnPoint_connection_conflict(root_pos, leaf_pos, my_graph)
                #     if res == 1:
                #         new_graph = Not_Orthogonal_TwoTurnPoint_connection(root, root_pos, leaf, leaf_pos, new_graph)
                #     else:
                #         update_graph(root, root_pos, leaf, leaf_pos, new_graph, res)
                # elif relative_pos == 3:
                #     cross_pos = (root_x, leaf_y)
                #     node = len(new_graph.node)
                #     new_graph.node = np.append(new_graph.node, node)
                #     new_graph.G.add_node(node)
                #     new_graph.G.add_edges_from([(root, node), (node, leaf)])
                #     new_graph.G.remove_edge(root, leaf)
                #     new_graph.color_map.append('yellow')
                #     new_graph.pos[node] = cross_pos
                #     None
                # elif relative_pos == 4:
                #     cross_pos = (root_x, leaf_y)
                #     node = len(new_graph.node)
                #     new_graph.node = np.append(new_graph.node, node)
                #     new_graph.G.add_node(node)
                #     new_graph.G.add_edges_from([(root, node), (node, leaf)])
                #     new_graph.G.remove_edge(root, leaf)
                #     new_graph.color_map.append('yellow')
                #     new_graph.pos[node] = cross_pos
                #     None
    for edge in my_graph.edge:
        # 对于每个根节点 连接出线开关
        node_a, node_b = edge[0], edge[1]
        # pos1,pos2是待处理的节点位置
        pos1, pos2 = new_graph.pos[node_a], new_graph.pos[node_b]

        if node_a in new_graph.root or node_b in new_graph.root:
            None
        # 不包含根节点，判断相对位置
        else:
            relation = relationship_between_root_and_leaf(pos1, pos2)
            # root1, root2 = find_root_from_two_leaf(new_graph, node_a, node_b)
            if relation == 0:
                # 如果正交关系
                # 判断直接连线是否冲突
                if whether_straight_connection_conflict(node_a, pos1, node_b, pos2, new_graph):
                    # 添加两个拐点连线
                    new_graph = Orthogonal_TwoTurnPoint_connection(node_a, pos1, node_b, pos2, new_graph)
                else:
                    # 直接连线
                    new_graph = update_graph(node_a, pos1, node_b, pos2, new_graph)
            else:
                # 如果不正交,
                res = whether_OneTurnPoint_connection_conflict(node_a, pos1, node_b, pos2, new_graph)
                if res == 1:  # 一个拐点都冲突
                    # 两个拐点连线
                    new_graph = Not_Orthogonal_TwoTurnPoint_connection(node_a, pos1, node_b, pos2, new_graph)

                else:
                    # 添加一个拐点res连线
                    new_graph = update_graph(node_a, pos1, node_b, pos2, new_graph, res)
    new_graph.show_graph()

def draw_topo_from_graph_v2(my_graph):
    '''
    1005针对线冲突需求，重新设计连线思路，严格分批次处理连线：
        1.root --- root_switch
        2.root_switch ---- middle
        3.middle --- contact_switch
    :param my_graph:
    :return:
    '''
    new_graph = copy.deepcopy(my_graph)
    new_graph = change_node_pos(new_graph)

    # root --- root_switch
    for root in new_graph.root:
        (x, y) = new_graph.pos[root]
        new_graph.pos[root] = (x - 0.2, y)
        pos_root = new_graph.pos[root]
        root_switch = np.argwhere(my_graph.adjacency_matrix[root] == 1).flatten()
        for leaf in root_switch:
            pos_leaf = new_graph.pos[leaf]
            if if_two_pos_orthogonal(new_graph.pos[root], new_graph.pos[leaf]):
                # 如果正交
                if whether_straight_connection_conflict(pos_root, pos_leaf, new_graph):
                    new_graph = Orthogonal_TwoTurnPoint_connection(root, pos_root, leaf, pos_leaf, new_graph)
                else:
                    # 直接相连
                    None
            else:
                # 如果不正交
                res = whether_OneTurnPoint_connection_conflict(pos_root, pos_leaf, new_graph)
                if res == 1:
                    new_graph = Not_Orthogonal_TwoTurnPoint_connection(root, pos_root, leaf, pos_leaf, new_graph)
                else:
                    update_graph(root, pos_root, leaf, pos_leaf, new_graph, res)
    # root_switch --- middle
