class SimpleTokenizer:
    def __init__(self, vocab_file):
        # 1. 读入 vocab.txt
        self.vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.vocab[token] = idx

        # 2. 建立 id 到 token 的反查表
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def tokenize(self, text):
        # 简化版：按空格切词，小写
        tokens = text.lower().split()
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(tok, self.vocab.get("[UNK]")) for tok in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i, "[UNK]") for i in ids]
