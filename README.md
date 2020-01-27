# Multi-label-Classifier-on-Privacy-Policies-
please make all commits in different branches

Added tokenizer, data cleaning and word2vec embedding functions
tokenizer(text:str) -> List[str]
usage:
form Data_preperation.tokenzie import tokenizer


text = "i love UCF"
tokenized = tokenizer(text)


get_embeddings usage:
form Data_preperation.Embeddngs import get_embeddings


model, embeddings = get_embeddings("word2vec",path_to_pretrained_word2vec,tokenized_text)



Please review and let me know about any changes
