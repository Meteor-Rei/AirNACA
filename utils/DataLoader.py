import numpy as np
import torch
from utils.tools import * 
import time
import random
import copy
from utils.tools import get_tag_list
from utils.caculation import build_dist_matrix_np
import os



selected_test_list = ["naca653618", "n2h15" , "n63010a", "n64212ma", "naca0012h", 
             "naca001034a08cli02", "naca23018", "naca633418", "naca651212a06",
             "nacacyh", "ncambre" ,  "n0009sm" , "n8h12"]


def _get_padded_coordinates(coordinates, padding_length):
    assert len(coordinates.shape) == 2

    padding = np.zeros((padding_length , coordinates.shape[1]))
    padding[0 : len(coordinates) , :] = coordinates
    return padding

def partition_selected(datas):
    train_list = []
    test_list  = []
    for data in datas:
        if data[0] in selected_test_list:
            test_list.append(data)
        else:
            train_list.append(data)
    return train_list , test_list

def partition_random(datas, ratio= 0.2):
    split_index = int(len(datas) * ratio)
    return datas[ : split_index] , datas[split_index : ] 



class BasicDataProcessor():
    def __init__(self , seed = None, type = "selected") -> None:
        self.data = []
        self.train_data = None
        self.test_data = None

        self.padding_data = []
        self.train_padding_data = None
        self.test_padding_data = None

        self.padding_distance_data = []
        self.train_padding_distance_data = None
        self.test_padding_distance_data = None

        self._load_data(type)

    def _load_data(self, type = "selected", padding_length = 135):
        for root, dirs, files in os.walk("./data/extracted"):  
            if len(files) == 0: 
                continue
            name = root.split("/")[-1]
            coordinates = np.loadtxt(root + "/" + "coordinates.txt")
            states = np.loadtxt(root + "/" + "states.txt")  
            length = len(coordinates)    
            self.data.append((name, length, coordinates, states))

            self.padding_data.append(
                (name, length, _get_padded_coordinates(coordinates, padding_length), states)
            )


            self.padding_distance_data.append(
                (name, length, _get_padded_coordinates(coordinates, padding_length), states, build_dist_matrix_np(coordinates, padding_length))
            )

        # 复用一下
        if type == "selected":
            self.train_data , self.test_data = partition_selected(self.data)
            self.train_padding_data, self.test_padding_data = partition_selected(self.padding_data)
            self.train_padding_distance_data, self.test_padding_distance_data = partition_selected(self.padding_distance_data)
        elif type == "random":
            self.train_data , self.test_data = partition_random(self.data)
            self.train_padding_data, self.test_padding_data = partition_random(self.padding_data)
            self.train_padding_distance_data, self.test_padding_distance_data = partition_random(self.padding_distance_data)



            
    def get_test_data(self):
        return self.test_data

    def get_train_data(self):
        return self.train_data

    def get_test_data_padding(self):
        return self.test_padding_data

    def get_train_data_padding(self):
        return self.train_padding_data

    def get_test_data_distance_padding(self):
        return self.test_padding_distance_data

    def get_train_data_distance_padding(self):
        return self.train_padding_distance_data





