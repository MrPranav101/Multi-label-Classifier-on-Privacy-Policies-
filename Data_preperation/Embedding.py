from gensim.models import KeyedVectors
import numpy as np

def get_embeddings(embedding_name,path_to_pretrained_model, doc):

    '''
    :param embedding_name: str, [word2vec]
    :param path_to_pretrained_model:  str, [path to pretrained embedding]
    :param doc: str, input text
    :return: embeddings of size len(doc) X 300
    '''
    if embedding_name == "word2vec":
        model = KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        return model,[model[token] if token in model else for token in doc else np.zeros(300)]
    if embedding_name == "fasttext":
        model = KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=False)
        return model,[model[token] if token in model else for token in doc else np.zeros(300)]
    if embedding_name == "Glove":
        with open(path_to_pretrained_model) as file:
            for line in file:
                split = line.split()
                model[split[0]] = split[1]
        return model,[model[token] if token in model else for token in doc else np.zeros(300)]
