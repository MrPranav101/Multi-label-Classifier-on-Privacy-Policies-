from gensim.models import KeyedVectors
import numpy as np


def get_embeddings(embedding_name,path_to_pretrained_model, doc):

    '''
    :param embedding_name: str, [word2vec]
    :param path_to_pretrained_model:  str, [path to pretrained embedding]
    :param doc: str, input text
    :return: model, embeddings of size len(doc) X 300
    '''

    if embedding_name == "word2vec":
        model = KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        return model, np.array([model[token] if token in model else np.zeros(300) for token in doc])
    if embedding_name == "fasttext":
        model = KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=False)
        return model, np.array([model[token] if token in model else np.zeros(300) for token in doc])
    if embedding_name == "glove":
        with open(path_to_pretrained_model) as file:
            for line in file:
                split = line.split()
                model[split[0]] = split[1]
        return model, np.array([model[token] if token in model else np.zeros(300) for token in doc])
