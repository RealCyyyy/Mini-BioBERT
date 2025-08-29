import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=256, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, attention_mask=None):
        bsz, seq_len, hidden_size = hidden_states.size()

        # Project Q, K, V
        q = self.query(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  
            scores = scores.masked_fill(mask == 0, -1e9)
    


        probs = self.softmax(scores)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)

        return self.out(context)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=256, num_heads=4, intermediate_size=512, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.gelu = nn.GELU()
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.out_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention + Residual
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_layer_norm(hidden_states + attn_output)

        # FFN + Residual
        ffn_output = self.output(self.gelu(self.intermediate(hidden_states)))
        ffn_output = self.output_dropout(ffn_output)
        hidden_states = self.out_layer_norm(hidden_states + ffn_output)

        return hidden_states
