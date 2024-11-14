'''
我还能逆转吗
'''
import argparse
import numpy as np

from exp.Exp_Normal import Exp_Normal
from utils.tools import calc_mse, plot_loss

parser = argparse.ArgumentParser(description='')

#Basic Setting 公公又用用啊
parser.add_argument('--model', type=str, default='transformer',help='model of experiment, options: [Coming soon]')
parser.add_argument('--epochs', type=int, default= 3000, help='training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--iter' ,type=int , default=10 , help="repeat times")
parser.add_argument('--flag' , type=str , default="normal", help="Experiment setting and folder name")
parser.add_argument('--fig', type=str, default= None, help="model name in the figure")
parser.add_argument('--device', type=str , default="gpu", help="load device")


args = parser.parse_args()
Exp = Exp_Normal

train_loss_list = []
vali_loss_list = []
pred_list = []
true_list = []

path = "./result/" + str(args.model) +"/"+str(args.flag) +"/"

# 我是神人吗? 我就想问问了
for  i in range(args.iter):
    args.seed = i
    exp = Exp(args)
    _ , train_loss , vali_loss , pred , true  = exp.train()
    train_loss_list.append(train_loss)
    vali_loss_list.append(vali_loss)
    pred_list.append(pred)
    true_list.append(true)

train_loss = np.mean(np.concatenate(train_loss_list).reshape(args.iter,-1),axis=0)
vali_loss  = np.mean(np.concatenate(vali_loss_list).reshape(args.iter,-1),axis=0)
pred = np.concatenate(pred_list)
true = np.concatenate(true_list)
mse = calc_mse(pred,true)

np.savetxt(path + "/pred_data.txt" , pred , fmt= "%.6f")
np.savetxt(path + "/true_data.txt" , true , fmt= "%.6f")
np.savetxt(path + "/real_loss.txt" , np.atleast_1d(mse),fmt="%.6f")