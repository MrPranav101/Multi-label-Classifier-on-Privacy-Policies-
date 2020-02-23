from gensim.models import KeyedVectors
import numpy as np


def get_embeddings(embedding_name, path_to_pretrained_model):

    '''
    :param embedding_name: str, [word2vec]
    :param path_to_pretrained_model:  str, [path to pretrained embedding]
    :return: model i.e. dictionary with embedding vocab and embeddings
    '''

    if embedding_name == "word2vec":
        model = KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=True)
        return model
    if embedding_name == "fasttext":
        model = KeyedVectors.load_word2vec_format(path_to_pretrained_model, binary=False)
        return model
    if embedding_name == "glove":
        with open(path_to_pretrained_model, encoding="utf-8") as file:
            model = {}
            for line in file:
                split = line.split()
                model[split[0]] = split[1]
        return model
