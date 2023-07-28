from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F


class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, attn_drop:float=0.1,
                 out_drop:float=0.1, bias:bool=True):
        super().__init__()
        # input dimension must be divisible by num_heads
        assert hidden_size % num_heads == 0
        # number of Attention heads
        self.nh = num_heads

        # linear layer to project queries, keys, values
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)

        # attention dropout layer to prevent overfitting
        self.attn_drop = nn.Dropout(attn_drop)

        # linear layer to project final output
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        # final output dropout layer to prevent overfitting
        self.out_drop = nn.Dropout(out_drop)

    # boolean `mask` of shape (batch_size, sequence_length)
    # where True is masked and False is unmasked
    def forward(self, x: Tensor, mask: BoolTensor|None = None):
        # batch size, sequence length, input dimension
        B, S, C = x.shape

        # split into queries, keys, & values of shape
        # batch size (B), num_heads (NH), sequence length (S), head size (HS)
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        # dot product queries and keys for each head
        # (B, NH, S, S) = (B, NH, S, HS) @ (B, NH, HS, S)
        attn = q @ k.transpose(-2, -1)

        # scale by square root of output dimension
        attn = attn / math.sqrt(k.size(-1))

        # reshape and mask attention scores
        if mask is not None:
            attn = attn.masked_fill(mask.view(B, 1, 1, S), float('-inf'))

        # apply softmax to get attention weights
        attn = attn.softmax(dim=-1)

        # apply dropout to attention weight
        attn = self.attn_drop(attn)

        # dot product attention weights with values of shape
        # (B, NH, S, HS) = (B, NH, S, S) @ (B, NH, HS, S)
        x = attn @ v

        # and transpose heads & sequence and reshape back to (B, S, C)
        x = x.transpose(1, 2).reshape(B, S, C)

        # apply final linear layer and dropout to get output (B, S, C)
        return self.out_drop(self.Wo(x))


class CausalAttention(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, block_size:int,
                 attn_drop:float=0.1, out_drop:float=0.1, bias:bool=True):
        super().__init__()
        # input dimension must be divisible by num_heads
        assert hidden_size % num_heads == 0
        # number of Attention heads
        self.nh = num_heads

        # linear layer to project queries, keys, values
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)

        # attention dropout layer to prevent overfitting
        self.attn_drop = nn.Dropout(attn_drop)

        # linear layer to project final output
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        # final output dropout layer to prevent overfitting
        self.out_drop = nn.Dropout(out_drop)

        # causal mask to ensure that Attention is not applied to future tokens where
        # block_size is the maximum sequence length of the transformer
        self.register_buffer('causal_mask',
            torch.triu(torch.ones([block_size, block_size], dtype=torch.bool), diagonal=1)
                .view(1, 1, block_size, block_size), persistent=False
        )

    # boolean `mask` of shape (batch_size, sequence_length)
    # where True is masked and False is unmasked
    def forward(self, x: Tensor, mask: BoolTensor|None = None):
        # batch size, sequence length, input dimension
        B, S, C = x.shape

        # split into queries, keys, & values of shape
        # batch size (B), num_heads (NH), sequence length (S), head size (HS)
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        # dot product queries and keys for each head
        # (B, NH, S, S) = (B, NH, S, HS) @ (B, NH, HS, S)
        attn = q @ k.transpose(-2, -1)

        # scale by square root of output dimension
        attn = attn / math.sqrt(k.size(-1))

        # apply input and causal mask
        combined_mask = self.causal_mask[:, :, :S, :S]
        if mask is not None:
            combined_mask += mask.view(B, 1, 1, S)
        attn = attn.masked_fill(combined_mask, float('-inf'))

        # apply softmax to get attention weights
        attn = attn.softmax(dim=-1)

        # apply dropout to attention weight
        attn = self.attn_drop(attn)

        # dot product attention weights with values of shape
        # (B, NH, S, HS) = (B, NH, S, S) @ (B, NH, HS, S)
        x = attn @ v

        # and transpose heads & sequence and reshape back to (B, S, C)
        x = x.transpose(1, 2).reshape(B, S, C)

        # apply final linear layer and dropout to get output (B, S, C)
        return self.out_drop(self.Wo(x))

class CausalCrossAttention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # input dimension must be divisible by num_heads
        assert hidden_size % num_heads == 0
        # number of Attention heads
        self.nh = num_heads

        # linear layer to project queries from decoder input
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)

        # linear layer to project keys and values from encoder output
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=bias)

        # attention dropout layer to prevent overfitting
        self.attn_drop = nn.Dropout(attn_drop)

        # linear layer to project final output
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        # final output dropout layer to prevent overfitting
        self.out_drop = nn.Dropout(out_drop)

        # causal mask to ensure that Attention is not applied to future tokens where
        # block_size is the maximum sequence length of the transformer
        self.register_buffer('causal_mask',
            torch.triu(torch.ones([block_size, block_size], dtype=torch.bool), diagonal=1)
                .view(1, 1, block_size, block_size), persistent=False
        )


    # boolean `mask` of shape (batch_size, sequence_length)
    # where True is masked and False is unmasked
    def forward(self, x: Tensor, y: Tensor, mask: BoolTensor|None = None):
        # batch size, sequence length, input dimension
        B, S, C = x.shape

        # split into queries of shape (B, NH, S, HS) from decoder input
        q = self.Wq(x).reshape(B, S, self.nh, C//self.nh).transpose(1, 2)

        # split into keys and values of shape (B, NH, S, HS) from encoder output
        y = self.Wkv(y).reshape(B, S, 2, self.nh, C//self.nh)
        k, v = y.transpose(3, 1).unbind(dim=2)

        # dot product queries and keys for each head
        # (B, NH, S, S) = (B, NH, S, HS) @ (B, NH, HS, S)
        attn = q @ k.transpose(-2, -1)

        # scale by square root of output dimension
        attn = attn / math.sqrt(k.size(-1))

        # apply input and causal mask
        combined_mask = self.causal_mask[:, :, :S, :S]
        if mask is not None:
            combined_mask += mask.view(B, 1, 1, S)
        attn = attn.masked_fill(combined_mask, float('-inf'))

        # apply softmax to get attention weights
        attn = attn.softmax(dim=-1)

        # apply dropout to attention weight
        attn = self.attn_drop(attn)

        # dot product attention weights with values of shape
        # (B,NH,S,S) @ (B,NH,S,HS) -> (B,NH,S,HS)
        x = attn @ v

        # and transpose heads & sequence and reshape back to (B,S,C)
        x = x.transpose(1, 2).reshape(B, S, C)

        # apply final linear layer and dropout to get output (B,S,C)
        return self.out_drop(self.Wo(x))