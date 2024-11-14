from exp.Exp_Basic import Exp_Basic
from utils.DataLoader import BasicDataProcessor
import torch.nn as nn
import torch
import torch.optim as optim
import utils.Log as Log

import numpy as np
import time
from utils.tools import calc_mse



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
            
        }
        model = None
        if self.args.model  ==  '':
            pass
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
    
    def _select_optimizer_scheduler(self, optim, gamma = 0.33):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim ,milestones=[4000,8000], gamma=0.4)
        return scheduler

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


    def _process_one_batch(self,data, lengths ,dist_matrix):
        # batch * sequence * feature , 嫩牛啊
        X_START = 0
        X_END = data.shape[-1] - 1
        Y_START = X_END
        Y_END = data.shape[-1]

        #截取feature部分，将输入和输出分开
        x = data[:,:,X_START:X_END]
        y = data[:,:,Y_START:Y_END]

        outputs, _ ,_ = self.model(x,lengths,dist_matrix)
        pred = []
        true = []   
        for i in range(outputs.shape[0]):
            pred.append(outputs[i,0:lengths[i],:])
            true.append(y[i,0:lengths[i],:])
        return torch.cat(pred),torch.cat(true)


    # weights 先等等吧 我人傻了
    def _process_one_batch_scores(self,data, lengths ,dist_matrix):
        # batch * sequence * feature , 嫩牛啊
        X_START = 0
        X_END = data.shape[-1] - 1
        Y_START = X_END
        Y_END = data.shape[-1]

        #截取feature部分，将输入和输出分开
        x = data[:,:,X_START:X_END]
        y = data[:,:,Y_START:Y_END]

        outputs, weight_and_output,scores  = self.model(x,lengths,dist_matrix)
        pred = []
        true = []   
        # weights_and_outputs = []

        score_list = [[] for _ in range(self.args.n_heads)]


        for i in range(outputs.shape[0]):
            pred.append(outputs[i,0:lengths[i],:])
            true.append(y[i,0:lengths[i],:])
            # weights_and_outputs.append(weight_and_output[i,0:lengths[i],:])

       
        for h in range(self.args.n_heads):
            for i in range(outputs.shape[0]):
                score_list[h].append(scores[i , h , 0 : lengths[i] , 0 : lengths[i]].detach().cpu().numpy())


        return torch.cat(pred),torch.cat(true),  None ,score_list
    
    def validate(self,criterion,vali_data, vali_length , vali_distance):
        batch_size = self.args.batch_size
        total_loss = []
        preds = []
        self.model.eval()

        with torch.no_grad():
            for i in range(0,vali_data.shape[0] // self.args.batch_size):
                cur_data = vali_data[i * batch_size : (i + 1) * batch_size, :, :]
                lengths = vali_length[i * batch_size : (i + 1) * batch_size]
                cur_distance_matrix = vali_distance[i * batch_size : (i + 1) * batch_size , : , : ]
                
                pred, true  = self._process_one_batch(cur_data,lengths,cur_distance_matrix)
                loss = criterion(pred.cpu(),true.cpu())
                total_loss.append(loss.item())
                preds.append(pred.detach().cpu().numpy())
            preds = np.concatenate(preds).reshape(-1)
        total_loss = np.average(total_loss)
        re_pred, re_true = self.data_processor.re_transform(preds)
        real_loss = calc_mse(re_pred,re_true)
        self.model.train()
        return total_loss , real_loss
    

    '''
        hhhh,劳资还能说什么呢 看下这些傻逼代码
        就嗯造 又变成屎山喽 
        哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦 张顺飞输咯
    '''
    def train_mcl(self, load_model_path = None):
        #先简单的处理一下数据把
        train_data_np, train_distance_np ,train_length_np = self.data_processor.get_train_data()
        test_data_np, test_distance_np ,test_length_np = self.data_processor.get_test_data()

        train_data = torch.from_numpy(train_data_np).float().to(self.device)
        train_length = torch.from_numpy(train_length_np).to(self.device)
        train_distance = torch.from_numpy(train_distance_np).float().to(self.device)

        test_data = torch.from_numpy(test_data_np).float().to(self.device)
        test_length = torch.from_numpy(test_length_np).to(self.device)
        test_distance = torch.from_numpy(test_distance_np).float().to(self.device)

        #经典准备工作
        batch_size = self.args.batch_size
        epochs = self.args.epochs
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        train_loss = []
        vali_loss = []
        re_preds = []
        re_trues = []

        for epoch in range(epochs):       
            train_loss_epoch = []
            self.model.train()
            for i in range(0,train_data.shape[0] // self.args.batch_size):
                cur_data = train_data[i*batch_size:(i+1)*batch_size, :, :]
                cur_distance_matrix = train_distance[i * batch_size : (i+1) * batch_size, :, :]
                lengths =  train_length[i * batch_size : (i + 1) * batch_size]


                #这里是执行第一次运算喵 固定coor, 训练stat网络
                self.model.only_coor_training()
                
                model_optim.zero_grad()
                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix) 
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()

                #这里是执行第二次运算喵 固定stat,  训练coor网络
                self.model.only_stat_training()

                model_optim.zero_grad()
                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix) 
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                train_loss_epoch.append(loss.item())

            #嚯嚯嚯，夸张哦
            train_loss_item = np.average(train_loss_epoch)
            vali_loss_item = self.validate(criterion,test_data,test_length, test_distance)

            train_loss.append(train_loss_item)
            vali_loss.append(vali_loss_item)

            #原来的代码 加个perf也是一坨屎 脑瘫设计 故意不去打印日志属于是what can i say
            if (epoch % 50) == 0: 
                Log.log_loss(self.args.model,self.args.aoa,epoch,train_loss_item,vali_loss_item)
            if epoch >= epochs - 50:
                re_pred , re_true, scores = self.test(test_data,test_length, test_distance)
                re_preds.extend(re_pred)
                re_trues.extend(re_true)
        self.save_file(train_loss,vali_loss,  np.array(re_preds) , np.array(re_trues))
        return self.model, train_loss, vali_loss, re_preds, re_trues
 

    #我还能逆转吗 我请问了 太傻逼了再搞个train类吧 为什么呀为什么呀
    def train_mtl_3times(self, load_model_path = None):
        #先简单的处理一下数据把
        train_data_np, train_distance_np ,train_length_np = self.data_processor.get_train_data()
        test_data_np, test_distance_np ,test_length_np = self.data_processor.get_test_data()

        train_data = torch.from_numpy(train_data_np).float().to(self.device)
        train_length = torch.from_numpy(train_length_np).to(self.device)
        train_distance = torch.from_numpy(train_distance_np).float().to(self.device)

        test_data = torch.from_numpy(test_data_np).float().to(self.device)
        test_length = torch.from_numpy(test_length_np).to(self.device)
        test_distance = torch.from_numpy(test_distance_np).float().to(self.device)

        #经典准备工作
        batch_size = self.args.batch_size
        epochs = self.args.epochs
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        train_loss = []
        vali_loss = []
        re_preds = []
        re_trues = []

        for epoch in range(epochs):       
            train_loss_epoch = []
            self.model.train()
            for i in range(0,train_data.shape[0] // self.args.batch_size):
                cur_data = train_data[i*batch_size:(i+1)*batch_size, :, :]
                cur_distance_matrix = train_distance[i * batch_size : (i+1) * batch_size, :, :]
                lengths =  train_length[i * batch_size : (i + 1) * batch_size]
                #版本又更新了 要踏马训练3次了 3次啊3次
                #g喽 哦 张顺飞输咯！

                

                

                #这里是执行第一次运算喵 固定context和function网络 训练processor网络
                model_optim.zero_grad()
                self.model.only_processor_training()
    
                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix)
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                
            
                #这里是执行第二次运算喵 固定function和processor, 训练context网络
                model_optim.zero_grad()
                self.model.only_context_training()
                
                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix)
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                

                #这里是执行第三次运算喵 固定context和processor 训练function网络
                model_optim.zero_grad()
                self.model.only_function_training()

                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix)
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                 
                train_loss_epoch.append(loss.item())

            #嚯嚯嚯，夸张哦
            train_loss_item = np.average(train_loss_epoch)
            vali_loss_item = self.validate(criterion,test_data,test_length, test_distance)

            train_loss.append(train_loss_item)
            vali_loss.append(vali_loss_item)

            #原来的代码 加个perf也是一坨屎 脑瘫设计 故意不去打印日志属于是what can i say
            if (epoch % 50) == 0: 
                Log.log_loss(self.args.model,self.args.aoa,epoch,train_loss_item,vali_loss_item)
            if epoch >= epochs - 50:
                re_pred , re_true = self.test(test_data,test_length, test_distance)
                re_preds.extend(re_pred)
                re_trues.extend(re_true)
        self.save_file(train_loss,vali_loss,  np.array(re_preds) , np.array(re_trues))
        return self.model, train_loss, vali_loss, re_preds, re_trues

    # 卧槽我是天天才 直接搞两个训练方式不就是ok了吗 虽然代码很冗余 但是可读性很好
    # 能问问我是神人吗
    def train_mtl(self, load_model_path = None):
        #先简单的处理一下数据把
        train_data_np, train_distance_np ,train_length_np = self.data_processor.get_train_data()
        test_data_np, test_distance_np ,test_length_np = self.data_processor.get_test_data()

        train_data = torch.from_numpy(train_data_np).float().to(self.device)
        train_length = torch.from_numpy(train_length_np).to(self.device)
        train_distance = torch.from_numpy(train_distance_np).float().to(self.device)

        test_data = torch.from_numpy(test_data_np).float().to(self.device)
        test_length = torch.from_numpy(test_length_np).to(self.device)
        test_distance = torch.from_numpy(test_distance_np).float().to(self.device)

        #经典准备工作
        batch_size = self.args.batch_size
        epochs = self.args.epochs
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        scheduler  = self._select_optimizer_cycle_scheduler(model_optim,4000)
        train_loss = []
        vali_loss = []
        re_preds = []
        re_trues = []
        weights = []
        test_mse_list= []

        for epoch in range(epochs):       
            train_loss_epoch = []
            self.model.train()
            for i in range(0,train_data.shape[0] // self.args.batch_size):
                cur_data = train_data[i*batch_size:(i+1)*batch_size, :, :]
                cur_distance_matrix = train_distance[i * batch_size : (i+1) * batch_size, :, :]
                lengths =  train_length[i * batch_size : (i + 1) * batch_size]



          
                # 安全
               
                # 这里是执行第一次运算喵 固定function, 训练context网络
                model_optim.zero_grad()
                self.model.only_context_training()
                
                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix) 
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                

                # 这里是执行第二次运算喵 固定context,  训练function网络
                model_optim.zero_grad()
                self.model.only_function_training()

                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix)
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                
                scheduler.step()
                train_loss_epoch.append(loss.item())

            #嚯嚯嚯，夸张哦
            train_loss_item = np.average(train_loss_epoch)
            vali_loss_item,real_loss = self.validate(criterion,test_data,test_length, test_distance)
            test_mse_list.append(real_loss)


            train_loss.append(train_loss_item)
            vali_loss.append(vali_loss_item)

            #原来的代码 加个perf也是一坨屎 脑瘫设计 故意不去打印日志属于是what can i say
            if (epoch % 50) == 0: 
                Log.log_loss(self.args.model,self.args.aoa,epoch,train_loss_item,vali_loss_item)
            if epoch >= epochs - 50:
                re_pred , re_true, weight,score_list  =  self.test(test_data,test_length, test_distance)
                re_preds.extend(re_pred)
                re_trues.extend(re_true)
                weights.append(weight)
        self.save_file(train_loss,vali_loss, np.array(re_preds), np.array(re_trues), score_list, test_mse= np.array(test_mse_list))
        # 把模型存下来吧 我存牛魔
        if self.args.save_model :
            torch.save(self.model.state_dict(), self.path + "/model.pt")
        return self.model, train_loss, vali_loss, re_preds, re_trues
 

    # 2887 locate train
    def train(self, load_model_path = None):
        #先简单的处理一下数据把
        train_data_np, train_distance_np ,train_length_np = self.data_processor.get_train_data()
        test_data_np, test_distance_np ,test_length_np = self.data_processor.get_test_data()

        train_data = torch.from_numpy(train_data_np).float().to(self.device)
        train_length = torch.from_numpy(train_length_np).to(self.device)
        train_distance = torch.from_numpy(train_distance_np).float().to(self.device)

        test_data = torch.from_numpy(test_data_np).float().to(self.device)
        test_length = torch.from_numpy(test_length_np).to(self.device)
        test_distance = torch.from_numpy(test_distance_np).float().to(self.device)

        #经典准备工作
        batch_size = self.args.batch_size
        epochs = self.args.epochs
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        scheduler  = self._select_optimizer_cycle_scheduler(model_optim,4000)

        train_loss = []
        vali_loss = []
        re_preds = []
        re_trues = []
        weights = []
        test_mse_list= []

        for epoch in range(epochs):       
            train_loss_epoch = []
            self.model.train()
            for i in range(0,train_data.shape[0] // self.args.batch_size):
                cur_data = train_data[i*batch_size:(i+1)*batch_size, :, :]
                cur_distance_matrix = train_distance[i * batch_size : (i+1) * batch_size, :, :]
                lengths =  train_length[i * batch_size : (i + 1) * batch_size]
            
                model_optim.zero_grad()
                pred,true  = self._process_one_batch(cur_data, lengths, cur_distance_matrix) 
                loss = criterion(pred.cpu(),true.cpu())
                loss.backward()
                model_optim.step()
                train_loss_epoch.append(loss.item())

            #嚯嚯嚯，夸张哦
            train_loss_item = np.average(train_loss_epoch)
            vali_loss_item,real_loss = self.validate(criterion,test_data,test_length, test_distance)
            test_mse_list.append(real_loss)

            train_loss.append(train_loss_item)
            vali_loss.append(vali_loss_item)
            scheduler.step()
            #原来的代码 加个perf也是一坨屎 脑瘫设计 故意不去打印日志属于是what can i say
            if (epoch % 50) == 0: 
                Log.log_loss(self.args.model,self.args.aoa,epoch,train_loss_item,vali_loss_item)
            if epoch >= epochs - 50:
                re_pred , re_true, weight,score_list  =  self.test(test_data,test_length, test_distance)
                re_preds.extend(re_pred)
                re_trues.extend(re_true)
                weights.append(weight)
        self.save_file(train_loss,vali_loss, np.array(re_preds), np.array(re_trues), score_list, test_mse= np.array(test_mse_list))
        # 把模型存下来吧 我存牛魔
        if self.args.save_model :
            torch.save(self.model.state_dict(), self.path + "/model.pt")
        return self.model, train_loss, vali_loss, re_preds, re_trues
    

    def test(self,test_data, test_length, test_distance):  
        batch_size = self.args.batch_size
        preds = []
        score_list = [[] for _ in range(self.args.n_heads)]
        weights = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0,test_data.shape[0] // self.args.batch_size):
                cur_data = test_data[i * batch_size : (i + 1) * batch_size,:,:]
                cur_distance_matrix = test_distance[i * batch_size : (i + 1) * batch_size,:,:]
                lengths = test_length[i * batch_size : (i + 1) * batch_size]
                pred , _ , weight, scores =  self._process_one_batch_scores(cur_data,lengths,cur_distance_matrix)
        
                preds.append(pred.detach().cpu().numpy())   
                # weights.append(weight.detach().cpu().numpy())
                for k in range(self.args.n_heads):
                    score_list[k].extend(scores[k])
        preds = np.concatenate(preds).reshape(-1)
        #计算损失
        re_pred , re_true = self.data_processor.re_transform(preds)
        #如果要求绘制就绘制,注意力分数
        self.model.train()
        return re_pred , re_true , None, score_list