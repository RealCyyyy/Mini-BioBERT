import torch
from modeling.bert_model import MiniBioBERT 

def test_model_forward():
    vocab_size = 30522  # 或者你自己的 vocab.txt 大小
    model = MiniBioBERT(vocab_size=vocab_size)

    input_ids = torch.randint(0, vocab_size, (2, 128))  # batch=2
    token_type_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)

    hidden_states, cls_output = model(input_ids, token_type_ids, attention_mask)
    assert hidden_states.shape == (2, 128, 256)
    assert cls_output.shape == (2, 256)
