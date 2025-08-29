import torch
import torch.nn as nn

class MLMHead(nn.Module):
    """Masked Language Modeling head"""
    def __init__(self, hidden_size, vocab_size, embedding_weights=None):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # 投影回词表大小
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        if embedding_weights is not None:
            # 权重共享：decoder.weight = word_embeddings.weight
            self.decoder.weight = embedding_weights

        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.act(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)  # (batch, seq_len, vocab_size)
        return logits


class NSPHead(nn.Module):
    """Next Sentence Prediction head"""
    def __init__(self, hidden_size):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)  # 二分类

    def forward(self, pooled_output):
        return self.classifier(pooled_output)  # (batch, 2)
