import torch
import torch.nn as nn
import numpy as np



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, inputs):

        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual) 

class PoswiseFeedForwardNoLayerNormNet(nn.Module):
    def __init__(self,d_model, d_ff):
        super(PoswiseFeedForwardNoLayerNormNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return output + residual 
