import torch.nn as nn
import torch
import numpy as np
from model.Transformer import Transformer






class Transformer_MLP(nn.Module):
    def __init__(self, d_model = 128, d_coordinate = 2,d_state = 2 , d_tgt = 4):
        super(Transformer_MLP, self).__init__()
        self.transformer = Transformer(d_model= d_model , d_src =d_coordinate)
        
        self.states_mlp = nn.Sequential(
            nn.Linear(d_state, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model , d_model),
            nn.GELU(),
            nn.Linear(d_model , d_tgt)
        )
    
    def forward(self, coordinates , length , states):
        geom, _  = self.transformer(coordinates, length)  # 1 * d_model
        state = self.states_mlp(states)  # n * d_model
        geom_expanded = geom.expand(state.shape[0] , -1)
        fusion_features = torch.cat((geom_expanded, state), dim = 1)
        result = self.fusion_mlp(fusion_features)
        return result



class MLP_Concat(nn.Module):
    def __init__(self, d_model = 128, d_coordinate = 2,d_state = 2 , d_tgt = 4):
        super(MLP_Concat, self).__init__()
        
        self.geom_mlp = nn.Sequential(
            nn.Linear(d_coordinate, d_model // 2),
            
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model),
            
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        self.states_mlp = nn.Sequential(
            nn.Linear(d_state, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model , d_model),

            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model , d_tgt)
        )
    
    def forward(self, coordinates , length, states):
        geom  = self.geom_mlp(coordinates)  # length * d_model
        geom  = geom[0 : length, :]
        geom = geom.mean(dim = 0 , keepdim = True)

        state = self.states_mlp(states)  # n * d_model

        geom_expanded = geom.expand(state.shape[0] , -1)
        fusion_features = torch.cat((geom_expanded, state), dim = -1)
        result = self.fusion_mlp(fusion_features)
        return result




