"""
# @FileName      : calculation
# @Time          : 2024-04-03 11:17:55
# @Author        : Meteor
# @description   : 原本的tool.py属于是什么屎都放进去了
    重新开一个吧
"""
import numpy as np
# 我心永恒
def _dist_calc_np(dist_list, start_idx , step):
    # 这里的n是存在的length
    n = len(dist_list)
    select_list = [0] * n
    for i in range(step):
        select_list[(i + start_idx) % n] = 1
    dist1 = 0.0
    dist2 = 0.0
    for i in range(n):
        if select_list[i] == 1 :
            dist1 = dist1 + dist_list[i]
        else:
            dist2 = dist2 + dist_list[i]  
    return min(dist1,dist2)



def build_dist_matrix_np(coordinates, padded_length):
    matrix = np.full((padded_length, padded_length),fill_value = -1.0e10)
    length = len(coordinates)
    dist_list = []

    #算出了挨着的点的距离
    for i in range(length):
        xz1 = coordinates[i, :]
        xz2 = coordinates[(i + 1) % length, :]
        dist = np.sqrt(np.sum(np.square(xz1 - xz2)))
        dist_list.append(dist)

    #第一层循环是batch_size
    for l in range(length):
        for k in range(length):
            if l == 0 :
                matrix[k][k] = 0.0
            else:
                matrix[k][(k+ l) % length] = _dist_calc_np(dist_list, k , l)
    return matrix


