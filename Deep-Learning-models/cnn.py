import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from gensim.models import KeyedVectors
import numpy as np
import sys
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import re
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(0, '../utils')

# from utils.Embedding import get_embeddings
# from utils.data_prep import data_clean

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

def regex(line):
    line = re.sub(r"\\[^\s]+", "", line)
    line = re.sub(r"\S*@\S*\s?", "", line)
    line = re.sub(r"https\S+", "", line)
    line = re.sub(r"www\S+", "", line)
    line = re.sub(r"[;/-:,\$\~\!\?\t\|\)\]\[\(\>\=\<\"\-\&\']", ' ', line)
    line = re.sub(' +', ' ', line)
    line = re.sub('\. \.', '.', line)
    line = re.sub(' \.', '.', line)
    line = re.sub('\.+', '.', line)
    return line


def data_clean(path_to_data, path_to_label):
    '''
    Inputs:
    path_to_data: path to data.txt
    path_to_label: path to label.txt
    Outputs:
    A pandas dataframe with the preprocessed data with the respective category labels
    '''
    data = []
    df = pd.read_excel(path_to_label)
    with open(path_to_data) as file:
        for line in file:
            line = regex(line)
            # to remove stopwords
            line = remove_stopwords(line)
            data.append(line.strip().lower())
            if data[-1] == "------------------------------------------------" \
                           "------------------------------------------------------":
                del data[-1]
    string = ""
    privacy_preprocessed = []
    for item in data[1:]:
        if item != data[0]:
            string += item
        else:
            privacy_preprocessed.append(string)
            string = ""
    privacy_preprocessed.append(string)

    df['Privacy_Policies'] = privacy_preprocessed
    df["len"] = df["Privacy_Policies"].apply(lambda x: len(x))
    df.drop(df[df["len"] == 0].index, inplace=True)

    return df

path_to_data = "../../dataset/data.txt"
path_to_label = "../../dataset/labels.xlsx"

df = data_clean(path_to_data, path_to_label)
# df.head()

path_to_embedding = "../../glove/glove.6B.300d.txt"
embedding_name = "glove"


embeddings_index = get_embeddings(embedding_name, path_to_embedding)

df["len"] = df["Privacy_Policies"].apply(lambda x: len(x))
df.drop(df[df["len"] == 0].index, inplace=True)
X, y = list(df["Privacy_Policies"]) , df.drop(["Privacy_Policies","Policy #","len"], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
del X,y

max_features = 100000
maxlen = 8000
embed_size = 300


tokenizer = text.Tokenizer(num_words=max_features,lower=True)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 8000

X_train = sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)


word_index = tokenizer.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

sequence_input = Input(shape=(maxlen, ))
x = Embedding(len(embedding_matrix), embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
convs=[]
filter_sizes = [2,3,4,5,6]
for filter_size in filter_sizes:
    l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(x)
    l_pool = GlobalMaxPooling1D()(l_conv)
    convs.append(l_pool)
l_merge = concatenate(convs, axis=1)
x = Dropout(0.1)(l_merge)  
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(9, activation="sigmoid")(x)
model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy'])


batch_size = 16
epochs = 4
X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=233)

filepath="../../models/cnn/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]

model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)
#model.load_weights(filepath)
print('Predicting....')
y_pred = model.predict(X_test,batch_size=32,verbose=1)

# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc

# # y_pred_keras = keras_model.predict(X_test).ravel()
# fpl = []
# tpl = []
# aucl = []
# for i in range(9):
#   fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,i], y_pred[:,i])
#   auc_keras = auc(fpr_keras, tpr_keras)
#   fpl.append(fpr_keras)
#   tpl.append(tpr_keras)
#   aucl.append(auc_keras)

# # fpr_keras = sum(fpl) #/len(fpl)
# # tpr_keras = sum(tpl)/len(tpl)
# # auc_keras =  sum(aucl)/len(aucl)
# for i in range(9):
#     fpr_keras, tpr_keras, auc_keras =  fpl[i], tpl[i], aucl[i]
#     from matplotlib import pyplot as plt
#     plt.figure(1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#     # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     plt.show()


# import numpy as np
# from scipy import interp
# import matplotlib.pyplot as plt
# from itertools import cycle
# from sklearn.metrics import roc_curve, auc

# n_classes = 9

# lw = 2

# y_score = y_pred

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])


# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# plt.figure(1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"red","green","yellow","orange","purple","brown"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,alpha = 0.3,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

n_classes = 9

lw = 2

y_score = y_pred

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"red","green","yellow","orange","purple","brown"])
classes = ['First Party Collection/Use', 'Third Party Sharing/Collection', 'User Choice/Control',
           'User Access, Edit and Deletion', 'Data Retention', 'Data Security', 'Policy Change ',
           'Do Not Track & Technology', 'International & Specific Audiences']
for i, cls,color in zip(range(n_classes), classes, colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,alpha = 0.3,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(cls, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="upper center", bbox_to_anchor=(1.565, 0.8))
plt.show()