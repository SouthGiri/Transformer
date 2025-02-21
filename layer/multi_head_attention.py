import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calculate_attention(query, key, value, mask):
    # query, key, value: (n_batch, seq_len, d_k)
    # mask: (n_batch, seq_len, seq_len)

    d_k = key.shape[-1]
    attention_score = torch.matmul(query, key.transpose(-2, -1))
    attention_score = attention_score / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    
    attention_prob = F.softmax(attention_score, dim=-1)
    out = torch.matmul(value, attention_prob) 

    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super().__init__()
        self.d_model = d_model
        self.h = h

        # d_embed, d_model
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)

        # d_model, d_embed
        self.out_fc = out_fc


    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len==n, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        
        n_batch = query.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h)
            out = out.transpose(1, 2)
            return out


        # (n_batch, h, seq_len, d_k)
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = calculate_attention(query, key, value, mask)
        out = out.transpose(1, 2)
        out = out.contiguous.view(n_batch, -1, self.d_model)
        out = self.out_fc(out)

        return out
