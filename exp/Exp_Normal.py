from exp.Exp_Basic import Exp_Basic
from utils.DataLoader import BasicDataProcessor
from utils import Log
from utils.tools import calc_mse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import os

from model.FinalFrame import Transformer_MLP, MLP_Concat




class Exp_Normal(Exp_Basic):
    def __init__(self, args):
        super(Exp_Normal, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'trans+mlp':Transformer_MLP,
            'mlp+concat': MLP_Concat,

        }
        model = None
        if self.args.model  ==  'trans+mlp':
            model = model_dict[self.args.model]().float() 
        elif self.args.model == "mlp+concat":
            model = model_dict[self.args.model]().float() 
        assert model != None
        return model
    
    def _build_data_loader(self):  
        return BasicDataProcessor()
        


    # Basic
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_optimizer_cycle_scheduler(self, optim, pre_epoch):
        def lr_lambda(epoch):
            if epoch < pre_epoch:
                return 1.0  # 前4000个epoch固定学习率
            else:
                cycle_epoch = epoch - pre_epoch
                max_lr = self.args.learning_rate
                min_lr = 0.00005
                cycle_length = 200  # 循环周期长度为200个epoch
                return min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(cycle_epoch % cycle_length / cycle_length * 3.141592653589793)))
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch(self, coordinates, length, states):
        x = states[: , 0 : 2]
        y = states[: , 2 : 6]
        pred = self.model(coordinates, length, x)
        return pred , y


    def _from_np2tensor(self, data_list):
        ret = []
        for name, length,coordinates, states in data_list:
            coordinates2 = torch.from_numpy(coordinates).float().to(self.device)
            states2 = torch.from_numpy(states).float().to(self.device)
            ret.append((name, length, coordinates2 , states2))
        return ret



    def validate(self,criterion, valid_data_list):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for name , length, coordinates, states in valid_data_list:
                pred, true  = self._process_one_batch(coordinates, length, states)
                loss = criterion(pred,true)
                total_loss.append(loss.item())
                preds.append(pred.cpu())
                trues.append(true.cpu())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
        total_loss = np.average(total_loss)
        real_loss = calc_mse(preds, trues)
        self.model.train()
        return total_loss , real_loss

    def test(self, test_data_list):
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for name , length, coordinates, states in test_data_list:
                pred,true  = self._process_one_batch(coordinates, length, states)
                preds.append(pred.cpu())
                trues.append(true.cpu())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
        real_loss = calc_mse(preds, trues)
        self.model.train()
        return preds, trues, real_loss

    def train(self):
        
        #先简单的处理一下数据把
        train_data_list  = self.data_processor.get_train_data_padding()
        test_data_list = self.data_processor.get_test_data_padding()

        train_data_tensor_list = self._from_np2tensor(train_data_list)
        test_data_tensor_list = self._from_np2tensor(test_data_list)

        epochs = self.args.epochs
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        train_loss = []
        vali_loss = []
        preds = []
        trues = []

        for epoch in range(epochs):       
            train_loss_epoch = []
            self.model.train()

            for name , length, coordinates, states in train_data_tensor_list:
                model_optim.zero_grad()
                pred, true  = self._process_one_batch(coordinates, length,states)
                loss = criterion(pred,true)
                train_loss_epoch.append(loss.item())
                loss.backward()
                model_optim.step()
                
            train_loss_item = np.average(train_loss_epoch)
            vali_loss_item , _  = self.validate(criterion, test_data_tensor_list)
            train_loss.append(train_loss_item)
            vali_loss.append(vali_loss_item)
            
            if (epoch % 50) == 0: 
                Log.log_loss(self.args.model,epoch,train_loss_item,vali_loss_item)
            if epoch >= epochs - 10:
                pred , true , _=  self.test(test_data_tensor_list)
                preds.extend(pred)
                trues.extend(true)
        
        preds = np.array(preds)
        trues = np.array(trues)

        self.save_file(train_loss, vali_loss,  preds, trues)
        return self.args.model , train_loss, vali_loss , preds , trues


    

       







        
