import os
import torch
import numpy as np
import time
import pickle
import utils.Log as Log
from utils.tools import calc_mse ,plot_loss, plot_attn_matrix
import matplotlib.pylab as plt
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self._build_store_path()
        self.data_processor = self._build_data_loader()
    
    
    def _build_store_path(self):
        #紧急更新，由于要整多进程并发，在time-label上再加个index 吧.
        time_label = time.strftime("%Y_%m_%d_%H_%M_%S") +"__" + str(self.args.seed)
        path = "./result/" + str(self.args.model) + "/" + str(self.args.flag) + "/" + time_label
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        self.path = path
    

    def _build_data_loader(self):
        raise NotImplementedError
        return None

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.device == "gpu":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else :
            return torch.device("cpu")

    def validate(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    

    def save_file(self,train_loss,vali_loss, re_preds , re_trues, score_list = None, test_mse = None, weights = None):
        #保存文件,保存的是50次的预测和真实
        real_loss = calc_mse(re_preds , re_trues)  
        Log.log_mse(self.args.model  , self.args.epochs, real_loss)
        np.savetxt(self.path + "/pred_data.txt" , re_preds , fmt= "%.6f")
        np.savetxt(self.path + "/true_data.txt" , re_trues , fmt= "%.6f")
        np.savetxt(self.path + "/real_loss.txt" , np.atleast_1d(real_loss),fmt="%.6f")
        np.savetxt(self.path + "/train_loss.txt" , train_loss , fmt="%.6f")
        np.savetxt(self.path + "/vali_loss.txt" , vali_loss , fmt="%.6f")
        if not test_mse is None:
            np.savetxt(self.path + "/test_mse.txt", test_mse, fmt="%.6f")
        if not weights is None:
            np.savetxt(self.path + "/weights.txt", weights, fmt="%.6f")

        self._plot_epochs(train_loss,vali_loss)
        if not score_list is None:
            with open(self.path + "/score_list.pkl", 'wb') as f:
                pickle.dump(score_list,f)
            self._plot_attn_figures(score_list)

    def _plot_attn_figures(self, score_list): 
        for i in range(self.args.n_heads):
            image_path = self.path + "/attn_figures/" + str(i) + "/"
            if os.path.exists(image_path):
                pass
            else:
                os.makedirs(image_path)
            for index , matrix in enumerate(score_list[i]):
                save_figure_file = f"attn_{index}.png"
                plot_attn_matrix(matrix,save_path= image_path + save_figure_file)

    # def plot_attn_map(self, score_list):



    def _plot_epochs(self,train_loss,vali_loss):  
        model_name = None
        if self.args.fig != None:
            model_name = self.args.fig
        else:
            model_name = self.args.model
        plot_loss(train_loss,vali_loss,[None,None,0,0.010] , "ff",
                self.path +"/loss.png")
        plot_loss(train_loss,vali_loss,[None,None,0,0.025] ,"ff",
                self.path +"/loss2.png")