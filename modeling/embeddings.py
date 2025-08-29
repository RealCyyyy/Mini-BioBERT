import torch
import torch.nn as nn

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, max_len=128, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        pos_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = (
            self.word_embeddings(input_ids) +
            self.position_embeddings(pos_ids) +
            self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(embeddings))
