# download_pubmed_corpus.py
from datasets import load_dataset
from tqdm import tqdm

def dump_pubmed_corpus(outfile="pubmed_corpus.txt", max_samples=None):
    """
    使用 HuggingFace 上可加载的 'scientific_papers' 数据集中的
    'pubmed' 子集抽取摘要写入文件（每行一篇）。
    """
    print("开始下载 hg 数据集 ‘scientific_papers’ 的 PubMed 子集…")
    ds = load_dataset("ccdv/pubmed-summarization", "section", split="train")

    count = 0
    with open(outfile, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc="写入抽象"):
            abstract = example.get("abstract", "")
            if not abstract:
                continue
            # 把换行压成一行
            cleaned = abstract.replace("\n", " ").strip()
            f.write(cleaned + "\n")
            count += 1
            if max_samples and count >= max_samples:
                break

    print(f"已写入 {count} 条摘要到 {outfile}")

if __name__ == "__main__":
    # 用 max_samples 控制测试输出数量，删除可导出全部
    dump_pubmed_corpus(outfile="pubmed_corpus.txt", max_samples=200000)
