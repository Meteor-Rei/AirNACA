import torch
import torch.nn as nn
import numpy as np

'''
重新审视一下 因为每个输入没有batch这个维度了
但是其实也可以手动的添加一个dimension 1 然后最后在吧删掉就行
那么这个部分 就都可以不用改了
'''

def get_attn_pad_mask(inputs, length, max_seq_length = None):
    # 但是实际上在这个数据集内都为1的batch size 还是加上吧
    batch_size = inputs.shape[0]

    if max_seq_length == None:
        max_seq_length = inputs.shape[1]
    mask_matrix = np.zeros([batch_size, max_seq_length, max_seq_length],dtype=bool)
    mask_matrix[: , :, length:] = True
    return torch.from_numpy(mask_matrix).to(inputs.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model , d_q , d_k , d_v , n_heads ):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_k  = d_k
        self.d_v =  d_v
        self.n_heads = n_heads
    
    #有mask和无mask的情况，这里直接重写两个方法算了
    def forward(self, input_Q, input_K, input_V, attn_mask = None):
        residual, batch_size = input_Q, input_Q.size(0)
        n_heads = self.n_heads
        d_k = self.d_k
        d_v = self.d_v
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2) 
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  
        if not attn_mask is None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context) 
        #这里探究一下layernorm和pe的关系 就先不整个tag来调了 直接搞
        return self.layer_norm(output + residual), attn
        # return output + residual, attn
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask = None):
        d_k = K.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if not attn_mask is None:
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn
    

class DistanceAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads) -> None:
        super(DistanceAttention, self).__init__()
        
        self.W_Q = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc  = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        # self.gelu = nn.GELU()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, input_Q, input_K,  input_V, dist_factor, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        n_heads = self.n_heads
        d_q = self.d_q
        d_k = self.d_k
        d_v = self.d_v
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_q).transpose(1,2) 
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 
        dist_factor = dist_factor.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = DistanceScaledDotAttention()(Q, K, V, dist_factor,attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context) 
        return output + residual, attn

class DistanceFlowedAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, n_heads) -> None:
        super(DistanceFlowedAttention, self).__init__()
        
        self.W_Q = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc  = nn.Linear(n_heads * d_v, d_model, bias=False)
        
        self.extra_info = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, input_Q, input_K,  input_V, inter_info,dist_factor, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        n_heads = self.n_heads
        d_q = self.d_q
        d_k = self.d_k
        d_v = self.d_v


        inter_coefficient = self.extra_info(inter_info)
   
        Q = ((inter_coefficient + 1.0) * self.W_Q(input_Q)).view(batch_size, -1, n_heads, d_q).transpose(1,2) 
        K = ((inter_coefficient + 1.0) * self.W_K(input_K)).view(batch_size, -1, n_heads, d_k).transpose(1,2)  
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 
        dist_factor = dist_factor.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = DistanceScaledDotAttention()(Q, K, V, dist_factor,attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context) 
        return output + residual, attn


class DistanceScaledDotAttention(nn.Module):
    def __init__(self) -> None:
        super(DistanceScaledDotAttention,self).__init__()

    def forward(self, Q, K, V, dist_factor, attn_mask):
        d_k = K.shape[-1]
        #这一步得到了注意力的scores，然后再乘一个矩阵的系数
        scores = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(d_k)
        scores = dist_factor * scores
        scores.masked_fill_(attn_mask,-1e10)
        attn = nn.Softmax(dim= -1)(scores)
        context = torch.matmul(attn,V)
        return context, attn


class AdditiveAttention(nn.Module):
    def __init__(self,  d_model, seq_length):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.Linear(d_model , seq_length, bias= False)
        self.W_k = nn.Linear(d_model , seq_length, bias= False)
        self.W_v = nn.Linear(d_model , d_model , bias= False)
        self.tanh = nn.Tanh()
        self.fc  = nn .Linear(d_model, d_model ,bias=False)
       
    
    def forward(self, query, key, value,mask):
        #将query扩展到与encoder_outputs相同的序列长度
        residual = value
        Q = self.W_q(query)
        K = self.W_k(key)
        scores = self.tanh(Q + K)
        V = self.W_v(value)
        
        scores = scores.masked_fill(mask, -1e10)
        attn = nn.Softmax(dim=-1)(scores) 
        context = torch.matmul(attn, V)
        return self.fc(context)  + residual, attn

        

