from Data_preperation.data_prep import data_clean, label_condense
import numpy as np
import string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

def feature_encoder(df):

    x_train, x_validation, y_train, y_validation = model_selection.train_test_split( \
                                                        df['Privacy_Policies'], \
                                                        df['Category_1'])
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_validation = encoder.fit_transform(y_validation)

    return x_train, x_validation, y_train, y_validation    

def word_embedder(x_train, x_val):
    ############# Count Vector as Feaature ###############
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(df['Privacy_Policies'])

    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(x_train)
    xvalid_count =  count_vect.transform(x_val)

    return xtrain_count, xvalid_count

def model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, y_val)
    

if __name__ == "__main__":
    
    path_to_data = 'dataset/data.txt'
    path_to_label = 'dataset/labels.xlsx'

    df = data_clean(path_to_data, path_to_label)
    # label_condense(df)
    x_train, x_val, y_train, y_val = feature_encoder(df)
    
    xtrain_count, xvalid_count = word_embedder(x_train, x_val)
    accuracy = model(naive_bayes.MultinomialNB(), xtrain_count, y_train, xvalid_count)

    print("NB, Count Vectors: ", accuracy)