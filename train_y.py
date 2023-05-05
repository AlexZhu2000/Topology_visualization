import numpy as np
import tensorflow as tf
chess_num = 81
def prepare():
    '''
    convert pos(x, y) to onehot
    :return:
    '''
    all_pos = np.load(open('3000_pos.npy', 'rb'))
    all_chessboard = []
    for i, pos  in enumerate(all_pos):
        # 对每一个图pos，有一个chess_map
        chess_map = np.zeros((30, chess_num))
        for n in range(30):
            # 对每个点 得到位置onehot_n
            (x, y) = pos[n]
            x = int(x)
            y = int(y)
            if y > 8:
                print('chessboard is small...')
                return
            onehot_n = (8 - y) * 9 + x
            chess_map[n][onehot_n] = 1
        all_chessboard.append(chess_map)
    all_chessboard = np.array(all_chessboard)
    np.save('3000_chessboard.npy', all_chessboard)
    return all_chessboard

if __name__ == '__main__':
    prepare()