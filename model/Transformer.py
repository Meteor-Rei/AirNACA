import torch
import torch.nn as nn
from model.Embed import PositionalEmbedding
from model.Attention import MultiHeadAttention, DistanceAttention, get_attn_pad_mask 
from model.MLP import PoswiseFeedForwardNet



# 不考虑距离 先跑一下吧
class EncoderLayer(nn.Module):
    def __init__(self,d_model , d_ff , d_q , d_k , d_v , n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model , d_q , d_k , d_v , n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model ,d_ff)
    def forward(self, inputs, mask):
        outputs, attn = self.attn(inputs, inputs, inputs, mask)
        outputs = self.pos_ffn(outputs)
        return outputs, attn
    
class TranformerEncoder(nn.Module):
    def __init__(self, d_model , d_ff , d_q , d_k ,d_v, n_layers , n_heads):
        super(TranformerEncoder, self).__init__()
        self.pos_emb = PositionalEmbedding(d_model, dropout=0.1)
        self.layers = nn.ModuleList([EncoderLayer(d_model ,d_ff , d_q, d_k,d_v ,n_heads) for _ in range(n_layers)])

    def forward(self, inputs, mask):
        outputs = self.pos_emb(inputs.transpose(0, 1)).transpose(0, 1) 
        attns = []
        for layer in self.layers:
            outputs, attn = layer(outputs, mask)
            attns.append(attn)
        return outputs, attns

class Transformer(nn.Module):
    def __init__(self,d_model=128,d_src=2, d_tgt=128,d_ff=512,
                d_q=16 ,d_k=16,d_v=16, n_layers =1 , n_heads = 8):
        super(Transformer, self).__init__()
        self.src_proj = nn.Linear(d_src, d_model,bias=False)
        self.encoder = TranformerEncoder(d_model, d_ff, d_q, d_k, d_v, n_layers, n_heads)
        #映射到最终结果的MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model , d_model // 2) ,
            nn.GELU(),
            nn.Linear(d_model // 2 , d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2 , d_model)
        )
    # 哎呀 为了兼容之前的代码 这里就加个batch维度吧
    def forward(self, inputs, length):
        inputs  = inputs.unsqueeze(0)
        outputs = self.src_proj(inputs)

        mask = get_attn_pad_mask(outputs, length)
        outputs, attns = self.encoder(outputs, mask)
        result = self.mlp(outputs)
        result = result[:, 0 : length, :]
        # 出来是一个 batch * length * d_model 的tensor
        result = result.mean(dim=1, keepdim=True)  # 保持 batch 维度，结果为 (1, d_model)
        return result.squeeze(0), attns[-1]
    

'''
=====================================================================================================
现在这里是Geometric 的部分， 涉及到distance attention那一套东西
'''


class GeometricEncoderLayer(nn.Module):
    def __init__(self, d_model = 128 ,d_ff = 512 , d_q = 16, d_k= 16, d_v = 16,n_heads = 8) -> None:
        super(GeometricEncoderLayer, self).__init__()
        self.attnMixer = DistanceAttention(d_model, d_q, d_k, d_v, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff) 

    def forward(self, geoms, distances, mask):
        outputs, score = self.attnMixer(geoms, geoms, geoms, distances, mask)
        outputs = self.ffn(outputs)
        return outputs,score



class GeometricModule(nn.Module):
    def __init__(self, d_src = 2, d_model = 128 , n_layers = 1,
                max_seq_length = 135, distance_weight = True):
        super(GeometricModule, self).__init__()
        self.src_proj = nn.Linear(d_src, d_model,bias=False)
        
        self.max_seq_length = max_seq_length
        self.dist_weight = nn.Parameter(torch.ones(max_seq_length,max_seq_length),requires_grad=True)\
            if distance_weight else nn.Parameter(torch.ones(max_seq_length,max_seq_length),requires_grad=False)

        self.layers = nn.ModuleList([GeometricEncoderLayer(d_model) for _ in range(n_layers)])

    
        #在注意力后可以选择是否使用mlp
        self.mlp = nn.Sequential(
            nn.Linear(d_model , d_model // 2) ,
            nn.GELU(),
            nn.Linear(d_model // 2 , d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2 , d_model)
        )
    
    '''
        这里attn的系数反正也一般只分析最后
    '''
    def forward(self, coordinates, length, distance):
        # 映射
        outputs = coordinates.unsqueeze(0)
        outputs = self.src_proj(outputs)

        distance = distance.unsqueeze(0)

        # mask
        attn_mask = get_attn_pad_mask(outputs, length)
        
        dist_factor = nn.Softmax(dim=-1)(distance)
        dist_factor = dist_factor * self.dist_weight
        attn = []

        # attention来喽
        for layer in self.layers:
            outputs , scores = layer(outputs, dist_factor, attn_mask)
        # outputs 可以选择是否在这里进行mlp啊

        outputs = outputs[:, 0 : length, :]
        # 出来是一个 batch * length * d_model 的tensor
        outputs = outputs.mean(dim=1, keepdim=True)  # 保持 batch 维度，结果为 (1, d_model)
        return outputs.squeeze(0), scores
