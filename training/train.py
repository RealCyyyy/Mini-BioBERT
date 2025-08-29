import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset

from modeling.bert_model import BertForPretraining


# ========== 参数 ==========
VOCAB_SIZE = 30522  # BERT vocab 大小（这里默认 WordPiece vocab）
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-5
WARMUP_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 简单 tokenizer（你可以替换成自己的 BERT tokenizer）==========
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ========== 数据处理函数 ==========
def encode_mlm_nsp(example):
    # 兼容 PubMed (abstract) 和 Wikitext (text)
    text = example.get("abstract") or example.get("text") or ""
    sentences = text.split(". ")

    if len(sentences) < 2:
        sentences = [text, ""]  # 保证至少有两句话

    # 随机挑选句子对
    import random
    if random.random() < 0.5:
        # 正样本：连续句子
        idx = random.randint(0, len(sentences) - 2)
        sent_a, sent_b = sentences[idx], sentences[idx + 1]
        nsp_label = 0  # IsNext
    else:
        # 负样本：随机拼接
        sent_a = random.choice(sentences)
        sent_b = random.choice(sentences)
        nsp_label = 1  # NotNext

    # 分词 & 编码
    encoding = tokenizer(
        sent_a,
        sent_b,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )

    # MLM：随机 mask
    input_ids = encoding["input_ids"].squeeze(0)
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    input_ids[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return {
        "input_ids": input_ids,
        "token_type_ids": encoding["token_type_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "mlm_labels": labels,
        "nsp_labels": torch.tensor(nsp_label),
    }


# ========== 加载数据 ==========
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
dataset = dataset.map(encode_mlm_nsp, remove_columns=dataset.column_names)
dataset.set_format(type="torch")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== 初始化模型 ==========
model = BertForPretraining(VOCAB_SIZE).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    return max(
        0.0,
        float(EPOCHS * len(dataloader) - current_step)
        / float(max(1, EPOCHS * len(dataloader) - WARMUP_STEPS)),
    )

scheduler = LambdaLR(optimizer, lr_lambda)

# ========== 训练 ==========
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_mlm_correct, total_mlm_total = 0, 0, 0
    total_nsp_correct, total_nsp_total = 0, 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            mlm_labels=batch["mlm_labels"],
            nsp_labels=batch["nsp_labels"]
        )

        loss = outputs["loss"]
        mlm_logits = outputs["mlm_logits"]
        nsp_logits = outputs["nsp_logits"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ---- 监控指标 ----
        mlm_preds = mlm_logits.argmax(dim=-1)
        mlm_mask = (batch["mlm_labels"] != -100)
        mlm_correct = (mlm_preds[mlm_mask] == batch["mlm_labels"][mlm_mask]).sum().item()
        mlm_total = mlm_mask.sum().item()

        nsp_preds = nsp_logits.argmax(dim=-1)
        nsp_correct = (nsp_preds == batch["nsp_labels"]).sum().item()
        nsp_total = batch["nsp_labels"].size(0)

        total_loss += loss.item()
        total_mlm_correct += mlm_correct
        total_mlm_total += mlm_total
        total_nsp_correct += nsp_correct
        total_nsp_total += nsp_total

        if step % 50 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}, "
                  f"MLM Acc {mlm_correct/mlm_total:.4f}, "
                  f"NSP Acc {nsp_correct/nsp_total:.4f}, "
                  f"LR {scheduler.get_last_lr()[0]:.6f}")

        global_step += 1

    print(f"Epoch {epoch} done. Avg Loss {total_loss/len(dataloader):.4f}, "
          f"MLM Acc {total_mlm_correct/total_mlm_total:.4f}, "
          f"NSP Acc {total_nsp_correct/total_nsp_total:.4f}")
