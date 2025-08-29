from tokenization.tokenizer import SimpleTokenizer
from data_pipeline.dataset_builder import DatasetBuilder

# 初始化 tokenizer （记得 vocab.txt 在 tokenization/ 目录下）
tokenizer = SimpleTokenizer("tokenization/vocab.txt")

# 初始化 dataset builder
builder = DatasetBuilder(tokenizer)

# 一个极小的语料
corpus = [
    "Cancer is a disease .",
    "It can affect humans .",
    "Dogs are animals .",
    "They like to play ."
]

# 构建训练样本
dataset = builder.build_from_corpus(corpus)

print("=== Training Samples ===")
for sample in dataset:
    print("\n--- One Sample ---")

    # 输入 ID
    print("input_ids:", sample["input_ids"])
    # 转回 tokens 方便看
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    print("tokens   :", tokens)

    # MLM 标签
    print("mlm_labels:", sample["mlm_labels"])

    # NSP 标签
    print("nsp_label:", sample["nsp_label"])
