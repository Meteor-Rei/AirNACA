'''
我还能逆转吗
'''
import argparse
import numpy as np

from exp.Exp_Distance import Exp_Distance
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
Exp = Exp_Distance

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
