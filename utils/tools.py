import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math


"""
    这个B文件现在真的是垃圾堆啊
    什么b函数都塞到这里了
"""


sequence_index = [[0,9],[9,17],[17,27],[27,39],[39,48],[48,58],[58,67],[67,77],[77,86],[86,95],
[95,118],[118,142],[142,165],[165,189],[189,217],[217,243],[243,266],[266,289],[289,312],[312,328],
[328,342],[342,357],[357,375],[375,391],[391,408],[408,425],[425,439],[439,450],[450,460],[460,470]]


# aoa_dict = {7:1880,12:3760,16:5640,18:7520,18.5:9400,19:11280,20:13160}


dist_attn_list = ["eudist","eudist_f","dist"]
basic_attn_list = ["basic","none","multihead"]

def get_tag_list():
    tags = np.zeros(470, dtype=int)
    for tag, (start, end) in enumerate(sequence_index):
        tags[start:end] = tag
    return tags



def min_max_normalize(data):
    # 遍历每一列
    for i in range(data.shape[1]):
        col = data[:, i]  # 获取第i列
        min_val = np.min(col)  # 计算最小值
        max_val = np.max(col)  # 计算最大值
        data[:, i] = (col - min_val) / (max_val - min_val)  # 归一化处理
    return data

def write_airfoil_txt():
    data_all = np.loadtxt("./datas/coor_status_cp_4230.txt")[0:470, [0,2]]
    for i, seq in enumerate(sequence_index):

        start = seq[0]
        end   = seq[1]
        airfoil = min_max_normalize(data_all[start:end, :])
        np.savetxt(f"./airfoils/airfoil_{i}.txt",airfoil,fmt="%.4f")
        







def equal_double(val1 , val2):
    return abs(val2 - val1) < 0.00001

def get_start_idx(data,aoa,col=3):
    for i in range(len(data)):
        if equal_double(data[i][col],aoa):
            return i
    raise ValueError("no matching index")




def split_data(data,start_idx,end_idx):
    test_data = data[start_idx:end_idx,:]
    part1 = data[0:start_idx,:]
    part2 = data[end_idx:len(data),:]
    train_data = np.concatenate([part1,part2],axis=0)
    return train_data,test_data

def build_single_sequence(data,max_seq_length , loop_load = False):
    if loop_load : 
        return build_loop_sequence(data, max_seq_length)

    feature = data.shape[-1]
    ret = np.zeros([max_seq_length,feature])
    row = data.shape[0]
    col = data.shape[1]
    ret[0:row,0:col] = data
    # top_half_index = min(max_seq_length , row + math.ceil(row / 2))
    # bottom_half_index = max(row , max_seq_length - math.ceil(row / 2))
    # ret[row : top_half_index , 0: col] = data[0: math.ceil(row/2),:] 
    # ret[bottom_half_index : max_seq_length , 0 :col] = data[row//2 : row, :]
    return ret



#这里就不写检查条件了，因为实际上都是
def build_loop_sequence(data,max_seq_length):
    feature = data.shape[-1]
    ret = np.zeros([max_seq_length,feature])
    row = data.shape[0]
    col = data.shape[1]
    ret[0:row,0:col] = data

    #这里相当于我们要在loop的接着的部分构建length / 2的上半部分
    row_2 = row // 2
    ret[row : row + row_2 , 0 : col] = data[0 : row_2 , 0 : col]
    ret[max_seq_length - row_2   : max_seq_length , 0 : col] = data[row - row_2 : row , 0 : col]
    return ret

def rebuild_from_sequence(data,lengths):
    ret = []
    assert data.shape[0] == len(lengths)
    for i in range(len(lengths)):
        seq = data[i]
        ret.append(seq[0:lengths[i],:])
    return np.concatenate(ret)


def build_long_piece(data,length,max_length):
    vector = np.zeros([7 * max_length])
    for i in range(length):
        vector[i*6 : i * 6 + 6] = data[i,0:6]
        vector[6 * max_length + i] = data[i , 6]
    return vector


def extend_list(data):
    ret = []
    for piece in data:
        ret.extend(piece[0])
    return np.array(ret)


def plot_attn_matrix(matrirx , title = None, figsize= (8,8), cmap='Reds', save_path= "nmsl.png"):
    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    pcm = ax.imshow(matrirx, cmap= cmap)
    fig.colorbar(pcm,ax=ax)
    plt.savefig(save_path)
    plt.close()

def plot_loss(train_loss,vali_loss, fig_axis,fig_title,save_path):
    plt.cla()
    plt.axis(fig_axis)
    x = None

    x = np.arange(1,len(train_loss) + 1)

    plt.plot(x , train_loss, c = 'blue' , label = "Train Loss")
    plt.plot(x , vali_loss , c = 'red' , label = "Test Loss")
    plt.title(fig_title,fontsize = 20)
    plt.legend()
    plt.savefig(save_path) 


def plot_hot_map(score,xyz,file_name, hero):
    plt.cla()
    fig = plt.figure(figsize=(12,10))
    ax  = fig.add_subplot(111,projection='3d')
    ax.title.set_text('Attention Map of %d' % (hero))
    norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
    im = ax.scatter3D(xyz[:,0], xyz[:,1] ,xyz[:,2] ,c= score ,s =15 ,cmap = 'jet',norm = norm)
    plt.colorbar(im,ax=ax,norm=norm,cmap = 'jet')
    plt.savefig(file_name)
    

def plot_hot_map_2d(score,xyz,file_name,hero):
    plt.clf()
    norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
    plt.title('Attention Map of %d' % (hero))
    plt.scatter(x = xyz[:,0] , y = xyz[:, 2], c= score ,s =15 ,cmap='jet', norm = norm )
    for i  in range(len(score)):
        plt.annotate(i , (xyz[i,0] ,xyz[i,2]))
    plt.colorbar(norm=norm , cmap = 'jet')
    plt.savefig(file_name)
    
        



def build_multi_sequence(data,max_seq_length,feature):
    ret = []
    for indices in sequence_index:
        start_idx = indices[0]
        end_idx = indices[1]
        seq_data = data[start_idx:end_idx,:]
        ret.append(build_multi_sequence(seq_data,max_seq_length,feature))
    return ret

def calc_mse(pred,true):
    return np.mean(np.square(pred - true))




class StandardScaler():
    def __init__(self,pcols = None):
        self.mean = 0.
        self.std = 1.
        self.cols = None
        self.fitted = False
        self.pcols = pcols
        

    def fit(self, data):
        if self.pcols is None:
            self.mean = data.mean(0)
            self.std = data.std(0)
            self.fitted = True
        else :
            p_data = data[:,self.pcols]
            self.mean = p_data.mean(0)
            self.std = p_data.std(0)
            self.fitted = True


    def transform(self, data):


        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if self.pcols is None:
            return (data - mean) / std
        else :
            p_data = data[:,self.pcols]
            p_res = (p_data - mean) / std
            data[: , self.pcols] = p_res
            return data 


    def inverse_transform(self, data):
        if not self.fitted:
            return data
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        
        if self.pcols is None:
            if data.shape[-1] != mean.shape[-1]:
                mean = mean[-1:]
                std = std[-1:]
            return (data * std) + mean
        else :
            p_data = data[:,self.pcols]
            if p_data.shape[-1] != mean.shape[-1]:
                mean = mean[-1:]
                std = std[-1:]
            p_res = (p_data * std) + mean
            data[ : , self.pcols] = p_res
            return data

def adjust_ax(ax):
    ax.set_xlim(1,2)
    ax.set_ylim(0,1.5)
    ax.set_zlim(-0.2,0.1)
    ax.grid(False)


def plot_3d_hot(data,file_name,fig_title,figsize=(12,12),norm_limit=(0,0.01)):
    plt.cla()
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111,projection = '3d')
    adjust_ax(ax)
    norm = matplotlib.colors.Normalize(vmin= norm_limit[0] , vmax = norm_limit[1])
    im = ax.scatter3D(data[:,0] , data[:,1] , data[:,2] , c = data[:,-1] , s =15 ,cmap='jet',norm = norm)
    
    plt.title(fig_title,fontsize = 20)
    plt.colorbar(im,ax=ax,norm=norm,cmap = 'jet')
    plt.savefig(file_name,dpi=400 )



def analyze_hot_pic(cps_pred :np.ndarray , cps_true : np.ndarray , row =470, col = 7 ,
    use_col = [0,1,2],file_name = "default2.png" , fig_title = "trans 18.5"):
    #这里的pred和true可能是50次test生成的，注意啦哈哈哈，那么就是 (470 * 50) * 7
    cp_pred = cps_pred.reshape(-1,row,col)[: ,: ,-1]
    cp_true = cps_true.reshape(-1,row,col)[: ,: ,-1]


    error = np.mean(np.square(cp_pred - cp_true),axis=0)
    error_var = np.var(error)
    print(error_var)
    final_data = np.concatenate([cps_true[0:row,use_col], error.reshape(-1,1)],axis = 1)
    plot_3d_hot(final_data ,file_name,fig_title)




   
    



