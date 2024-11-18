from exp.Exp_Basic import Exp_Basic
from utils.DataLoader import BasicDataProcessor
import torch.nn as nn
import torch
import torch.optim as optim
import utils.Log as Log

import numpy as np
import time
from utils.tools import calc_mse
from model.FinalFrame import GeometricConcat



class Exp_Distance(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        # 我能问问你吗，需不需要复现 需要复现还得配合data那边的种子 虽然data那边是按照seed来shuffle的
        # if type(args.seed) == int: 
        #     torch.manual_seed(args.seed)
        # else:
        #     torch.manual_seed(int(time.time()))
    
    def _build_model(self):
        model_dict = {
            "geom+concat" : GeometricConcat,

        }
        model = None
        if self.args.model  ==  "geom+concat":
            model = model_dict[self.args.model]().float() 
        elif self.args.model == '':
            pass

        assert model != None
        return model
    
    
    def _build_data_loader(self):  
        return BasicDataProcessor()
    
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        #模型的最终输出的是 batch * max_seq_length * 1
        criterion = nn.MSELoss()
        return criterion
    
    def _process_one_batch(self, coordinates, length, states, distances):
        x = states[: , 0 : 2]
        y = states[: , 2 : 6]
        pred = self.model(coordinates, length, x, distances)
        return pred , y


    # 这里要改改哦  因为有distances 了
    def _from_np2tensor(self, data_list):
        ret = []
        for name, length, coordinates, states, distances in data_list:
            coordinates2 = torch.from_numpy(coordinates).float().to(self.device)
            states2 = torch.from_numpy(states).float().to(self.device)
            distances2 = torch.from_numpy(distances).float().to(self.device)
            ret.append((name, length, coordinates2 , states2, distances2))
        return ret


    def validate(self,criterion, valid_data_list):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for name , length, coordinates, states, distances in valid_data_list:
                pred, true  = self._process_one_batch(coordinates, length, states, distances)
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
            for name , length, coordinates, states, distances in test_data_list:
                pred,true  = self._process_one_batch(coordinates, length, states, distances)
                preds.append(pred.cpu())
                trues.append(true.cpu())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
        real_loss = calc_mse(preds, trues)
        self.model.train()
        return preds, trues, real_loss


    def train(self):
        #先简单的处理一下数据把
        train_data_list  = self.data_processor.get_train_data_distance_padding()
        test_data_list = self.data_processor.get_test_data_distance_padding()

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

            for name , length, coordinates, states, distances in train_data_tensor_list:
                model_optim.zero_grad()
                pred, true  = self._process_one_batch(coordinates, length,states, distances)
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

    

