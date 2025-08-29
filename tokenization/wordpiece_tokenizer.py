import re

class WordPieceTokenizer:
    def __init__(self, vocab_file, unk_token="[UNK]"):
        # 加载词表
        self.token2id = {}
        self.id2token = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                self.token2id[token] = i
                self.id2token.append(token)

        self.unk_token = unk_token

    def basic_tokenize(self, text):
        """
        基础切分：小写化 + 按标点和空格切
        """
        text = text.lower()
        # 在标点前后加空格，再 split
        text = re.sub(r"([.,!?;])", r" \1 ", text)
        return text.strip().split()

    def wordpiece_tokenize(self, word):
        """
        对单个词做 WordPiece 分词
        """
        if word in self.token2id:
            return [word]

        sub_tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            # 从长到短匹配
            while start < end:
                substr = word[start:end]
                if start > 0:  # 不是词首就加 "##"
                    substr = "##" + substr
                if substr in self.token2id:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return [self.unk_token]
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def tokenize(self, text):
        """
        完整分词流程：先基础切分，再 WordPiece
        """
        split_tokens = []
        for word in self.basic_tokenize(text):
            split_tokens.extend(self.wordpiece_tokenize(word))
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id2token[i] for i in ids]
