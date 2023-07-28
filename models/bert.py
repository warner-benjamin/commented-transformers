"""
bert.py is a highly commented implementation of a modern BERT encoder Transformer

The codebase for bert.py is inspired by:
nanoGPT https://github.com/karpathy/nanoGPT - Copyright (c) 2022 Andrej Karpathy - MIT License
cramming https://github.com/JonasGeiping/cramming - Copyright (c) 2022 Jonas Geiping - MIT License
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, context_size: int, hidden_size: int):
        super().__init__()
        # create the positional encoding tensor of shape
        # maximum sequence length (MS) by embedding dimension (C)
        pe = torch.zeros(context_size, hidden_size, dtype=torch.float)

        # pre-populate the position and the div_terms
        position = torch.arange(context_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000) / context_size)
        )

        # even positional encodings use sine, odd cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as a buffer so autograd doesn't modify
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor):
        # return the pre-calculated positional encodings
        # up to sequence length (S). output shape (1, S, C)
        return self.pe[:, :x.shape[1], :]


class FeedForward(nn.Module):
    def __init__(self, hidden_size:int, expand_size:int, act:nn.Module=nn.GELU,
                 drop:float=0.1, bias:bool=True):
        super().__init__()
        # project input to expanded dimension
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)

        # activation function to introduce non-linearity
        self.act = act()

        # project back to the input dimension
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)

        # optional dropout layer to prevent overfitting
        self.drop = nn.Dropout(drop)

    def forward(self, x:Tensor):
        x = self.fc1(x) # apply first linear layer
        x = self.act(x) # apply activation function
        x = self.fc2(x) # apply second linear layer
        x = self.drop(x) # optionally apply dropout layer
        return x


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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, expand_size:int,
                 attention:nn.Module=BidirectionalAttention, act:nn.Module=nn.GELU,
                 attn_drop:float=0.1, out_drop:float=0.1, ffn_drop:float=0.1,
                 bias:bool=True):
        super().__init__()
        # first pre-norm layer
        self.norm1 = nn.LayerNorm(hidden_size)
        # initialize the attention layer
        self.attn = attention(
            hidden_size=hidden_size, num_heads=num_heads, attn_drop=attn_drop,
            out_drop=out_drop, bias=bias
        )

        # second pre-norm layer
        self.norm2 = nn.LayerNorm(hidden_size)
        # initialize the feed forward network (MLP)
        self.ffn = FeedForward(
            hidden_size=hidden_size, expand_size=expand_size, act=act,
            drop=ffn_drop, bias=bias,
        )

    def forward(self, x: Tensor):
        # normalize input then add residual to attention output
        x = x + self.attn(self.norm1(x))

        # normalize input then add residual to feedforward output
        return x + self.ffn(self.norm2(x))


class BERT(nn.Module):
    def __init__(self, num_layers:int, vocab_size:int, hidden_size:int, num_heads:int,
                 context_size:int, expand_size:int, attention:nn.Module=BidirectionalAttention,
                 act:nn.Module=nn.GELU, embed_drop:float=0.1, attn_drop:float=0.1,
                 out_drop:float=0.1, ffn_drop:float=0.1, head_norm:bool=True,
                 tie_weights:bool=True, head_bias:bool=True, bias:bool=True):
        super().__init__()
        # initialize vocab & positional encodings to convert numericalied tokens
        # & position indicies to token and position vectors, with optional dropout
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_encode = PositionalEncoding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        # initialize num_layers of transformer layers
        self.tfm_blocks = nn.ModuleList([TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads, expand_size=expand_size,
                attention=attention, act=act, bias=bias, attn_drop=attn_drop,
                out_drop=out_drop, ffn_drop=ffn_drop)
            for _ in range(num_layers)])

        # optional pre-head normalization
        self.head_norm = nn.LayerNorm(hidden_size) if head_norm else nn.Identity()

        # predicts the next token in the sequence
        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)

        # optionally set the vocab embedding and prediction head to share weights
        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        self.apply(self._init_weights)

    def forward(self, x: Tensor, return_preds:bool=True):
        # convert numericalized tokens of shape (B, S)
        # into token embeddings of shape (B, S, C)
        tokens = self.vocab_embed(x)
        # positional encodings are shape (S, C)
        pos = self.pos_encode(x)

        # positional encodings are added to token embeddings
        x = self.embed_drop(tokens + pos)

        # pass token vectors through all transformer layers
        for block in self.tfm_blocks:
            x = block(x)

        # apply optional pre-head normalization
        x = self.head_norm(x)

        # if MLM pretraining, don't predict outputs here
        if return_preds:
            # converts input token vectors of shape (B, S, C) to probability
            # distribution of shape batch, sequence length, vocabulary size (B, S, VS)
            return self.head(x)
        else:
            return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class BERTForMaskedLM(BERT):
    def __init__(self, loss_fn:nn.Module=nn.CrossEntropyLoss(),
                 mlm_prob:float|None=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.mlm_prob = mlm_prob

    def forward(self, input_ids: Tensor, labels: Tensor, mlm_prob: float|None = None):
        x = super().forward(input_ids, False)

        # flatten both the labels and the intermediate outputs
        labels = labels.view(-1)
        x = x.view(labels.shape[0], -1)

        # only select the masked tokens for predictions
        mask_tokens = labels != self.loss_fn.ignore_index

        # torch.compile with fullgraph cannot have dynamic shapes
        # if `mlm_prob` is set, this will create workable indicies
        # if `mlm_prob` is None, then fullgraph=True cannot be used
        mlm_prob = self.mlm_prob if mlm_prob is None else mlm_prob
        if mlm_prob is not None:
            num_masks = math.floor(self.mlm_prob * mask_tokens.shape[0])
        else:
            num_masks = mask_tokens.sum().int()
        indices = torch.argsort(mask_tokens.int())[-num_masks:]

        # selecting the masked tokens reshapes x to (B*S, VS) and labels to (B*S)
        x = x[indices]
        labels = labels[indices]

        # converts input token vectors of shape (B*S, C)
        # to probability distribution of shape (B*S, VS)
        logits = self.head(x)

        # return both the logits and the loss
        return {'logits': logits, 'loss': self.loss_fn(logits, labels)}