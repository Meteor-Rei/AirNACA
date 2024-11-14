# 记录
emmm， 距离上次搞有点久远了，所以记录一下。
目前这个数据集中的数据格式与DLR-F11里的数据具有一点不同

DLR-F11 是  batch * (samples + paddings) * dimension(coordinate + status)



而现在的NACA数据可以认为是每个飞机上的数据是一个batch，一个batch内有以下的文件
机翼的坐标(最大的数量是131) + 飞行状态参数 + (坐标间的距离)






