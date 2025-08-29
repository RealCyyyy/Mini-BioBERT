from tokenization.tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer("tokenization/vocab.txt")

text = "The patient has lung cancer."
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", ids)

back = tokenizer.convert_ids_to_tokens(ids)
print("Back to tokens:", back)
