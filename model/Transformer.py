import torch.nn as nn
from model.Embed import PositionalEmbedding
from model.Attention import MultiHeadAttention, get_attn_pad_mask 
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