from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from utils.data_prep import data_clean
from utils.eval import eval_auc

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

maxlen = 512
batch_size = 8192
nb_epoch = 100
train_len = 400

def split(df, label_cols):

    x_train = df['Privacy_Policies'][:train_len]
    x_validation = df['Privacy_Policies'][train_len:]
    
    y_train_dict = dict(); y_test_dict = dict()
    for i in label_cols:
        y_train_dict[i] = df[i][:train_len].values
        y_test_dict[i] = df[i][train_len:].values

    return x_train, x_validation, y_train_dict, y_test_dict

def tokenization(X_train, X_test, path_to_embedding):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embeddings_dictionary = dict()
    glove_file = open(path_to_embedding, encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return X_train, X_test, vocab_size, embedding_matrix

def train(X_train, X_test, y_train_dict, y_test_dict, embedding_matrix, vocab_size, trainable=False):
    input_1 = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input_1)
    LSTM_Layer1 = LSTM(128)(embedding_layer)

    output1 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output2 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output3 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output4 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output5 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output6 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output7 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output8 = Dense(1, activation='sigmoid')(LSTM_Layer1)
    output9 = Dense(1, activation='sigmoid')(LSTM_Layer1)

    model = Model(inputs=input_1, outputs=[output1, output2, output3, output4, output5, output6, output7, output8, output9])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    if trainable:
        history = model.fit(x=X_train, \
                        y=[y_train_dict[value] for value in y_train_dict], \
                        batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2)

        model.save('LSTM.h5')
        score = model.evaluate(x=X_test, y=[y_test_dict[value] for value in y_test_dict], verbose=1)
    
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        return score
    
    else:
        model.load_weights('Deep-Learning-models/LSTM.h5')
        
        prediction = model.predict(X_test)
        return prediction


if __name__ == "__main__":
    
    path_to_data = 'dataset/data.txt'
    path_to_label = 'dataset/labels.xlsx'
    path_to_embedding = 'glove.6B.100d.txt'
    
    df = data_clean(path_to_data, path_to_label)
    label_cols = ['Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5', 'Category_6', 'Category_7', 'Category_8', 'Category_9']

    X_train, X_val, y_train_dict, y_val_dict = split(df, label_cols)
    y_true = np.array(list(y_val_dict.values()))

    X_train, X_val, vocab_size, embedding_matrix = tokenization(X_train, X_val, path_to_embedding)
    y_score = train(X_train, X_val, y_train_dict, y_val_dict, embedding_matrix, vocab_size)
    
    eval_auc(np.resize(np.array(y_score), (9, 52)).T, np.asarray(y_true).T)