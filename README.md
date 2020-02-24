# Multi-label-Classifier-on-Privacy-Policies-
please make all commits in different branches

Added tokenizer, data cleaning and word2vec embedding functions


tokenizer(text:str) -> List[str]

usage:

import sys
sys.path.insert(1, '../utils')

from utils.data_prep import data_clean


text = "i love UCF"

tokenized = tokenizer(text)


get_embeddings usage:


import sys
sys.path.insert(1, '../utils')

from utils.Embeddings import get_embeddings


model = get_embeddings("word2vec",path_to_pretrained_word2vec,tokenized_text)


evaluation method is coming soon.

Please review and let me know about any changes


TO DO
ADD BPEmb
ADD TAA https://github.com/PavelOstyakov/toxic/blob/master/tools/extend_dataset.py
