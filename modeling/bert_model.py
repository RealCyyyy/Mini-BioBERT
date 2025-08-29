# modeling/bert_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, attention_mask=None):
        # ✅ 转换 mask：HF 是 1=keep, 0=pad；PyTorch MHA 是 True=mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (batch, seq_len), dtype=bool

        attn_output, _ = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x


class MiniBERT(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_act = nn.Tanh()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )

        x = self.norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        pooled_output = self.pooler_act(self.pooler(x[:, 0]))
        return x, pooled_output


class BertForPretraining(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = MiniBERT(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.nsp_head = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        return mlm_logits, nsp_logits

