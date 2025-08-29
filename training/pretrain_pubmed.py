import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import os
import random
from modeling.bert_model import BertForPretraining, BertConfig
from tqdm import tqdm  # 新增

# =====================
# 配置
# =====================
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LR = 5e-5
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 数据集 (PubMed 语料)
# =====================
class PubMedDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        # 简单切分成两个句子 (模拟 NSP)
        if "." in line:
            parts = line.split(".")
            sent_a = parts[0]
            sent_b = parts[1] if len(parts) > 1 else parts[0]
        else:
            sent_a, sent_b = line, line

        # 50% 正例 / 50% 负例
        if random.random() < 0.5:
            is_next = 1
        else:
            rand_idx = random.randint(0, len(self.lines) - 1)
            sent_b = self.lines[rand_idx]
            is_next = 0

        encoding = self.tokenizer(
            sent_a,
            sent_b,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        # 生成 MLM 标签
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
        labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 仅计算 mask 位置的 loss

        # 80% 替换 [MASK]，10% 随机词，10% 保持不变
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, attention_mask, token_type_ids, labels, torch.tensor(is_next)


# =====================
# collate_fn
# =====================
def collate_fn(batch):
    input_ids, attention_mask, token_type_ids, labels, is_next = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(attention_mask),
        torch.stack(token_type_ids),
        torch.stack(labels),
        torch.stack(is_next),
    )


# =====================
# 训练循环
# =====================
def train():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = PubMedDataset("pubmed_corpus.txt", tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    model = BertForPretraining(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fn = nn.CrossEntropyLoss()

    # 断点续训
    start_epoch = 0
    ckpt_path = os.path.join(SAVE_DIR, "latest.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            input_ids, attention_mask, token_type_ids, labels, is_next = [x.to(device) for x in batch]

            mlm_logits, nsp_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            mlm_loss = mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))
            nsp_loss = nsp_loss_fn(nsp_logits.view(-1, 2), is_next.view(-1))
            loss = mlm_loss + nsp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每步都显示进度条信息
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "MLM": f"{mlm_loss.item():.4f}",
                "NSP": f"{nsp_loss.item():.4f}"
            })

        # 每个 epoch 保存 checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, ckpt_path)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"biobert_pretrained_pubmed_epoch{epoch}.pt"))

    print("Training finished!")


if __name__ == "__main__":
    train()
    train()
