import torch
import torch.nn as nn
import math



class InputEmedding(nn.Module):

    def __init__(self,d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model**0.5)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_lenght = seq_length
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape(seq_lenght, d_model)
        pe = torch.zeros(seq_length, d_model)

        # create a vector lof shape(seq_lenght, 1)
        position = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        # apply the sin to even posititons
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_lenght, d_model)

        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        std = x.std(dim=-1, keepdim = True)
        mean = x.mean(dim=-1, keepdim = True)
        return self.alpha((x-mean) / (std + self.eps)) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(self.d_model, self.d_ff)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        relu = torch.relu(self.linear_1(x))
        drop = self.dropout(relu)
        return self.linear_2(drop)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.h = heads
        self.d_k = d_model // heads
        self.w_k = nn.Linear(d_model, d_model) #(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model) #(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model) #(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model) #(d_model, d_model)

    @staticmethod
    def attention(keys, queries, values, mask, dropout):
        # (batch, heads, seq_length, d_k) --> (batch, heads, seq_length, seq_length)
        attention_scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -torch.inf)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ values), attention_scores


    def forward(self, q, k, v, mask):
        key   = self.w_k(k) #(batch, seq_length, d_model) * (d_model, d_model ) --> (batch, seq_length, d_model)
        query = self.w_q(q) #(batch, seq_length, d_model) * (d_model, d_model ) --> (batch, seq_length, d_model)
        value = self.w_v(v) #(batch, seq_length, d_model) * (d_model, d_model ) --> (batch, seq_length, d_model)

        keys = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_length, heads, d_k) --> (batch, heads, seq_length, d_k) in transposition
        queries = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_length, heads, d_k) --> (batch, heads, seq_length, d_k) in transposition
        values = value.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # (batch, seq_length, heads, d_k) --> (batch, heads, seq_length, d_k) in transposition

        x, self.attention_scores = MultiHeadAttention.attention(keys, queries, values, mask, self.dropout)

        # (batch, heads, seq_length, d_k) --> (batch, seq_length, heads, d_k) --> (batch, seq_length, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)
    

# ye wala --> 
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    


class EncoderBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention
        self.feed_forward = FeedForwardBlock
        self.dropout = dropout
        self.residual_connection = nn.ModuleList([ResidualConnection(self.dropout) for _ in range(2)])

    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward(x))
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers):
        self.layers = nn.ModuleList
        self.norm = LayerNormalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
