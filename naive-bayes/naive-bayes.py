from Data_preperation.data_prep import data_clean
from Data_preperation.eval import eval_auc
import numpy as np
import string
import pickle
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import ensemble

def feature_encoder(df, category):

    x_train, x_validation, y_train, y_validation = model_selection.train_test_split( \
                                                        df['Privacy_Policies'], \
                                                        df[category])
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

def model(feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    clf = naive_bayes.MultinomialNB()
    clf.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = clf.predict(feature_vector_valid)
    
    return clf, metrics.accuracy_score(predictions, y_val), predictions
    
def save_pkl(name, clf, outdir):
    f = open(outdir + name + '.pkl', 'wb')
    pickle.dump(clf, f)
    f.close()

def load_pkl(name, outdir):
    f = open(outdir + name + '.pkl', 'rb')
    clf = pickle.load(f)
    f.close()



if __name__ == "__main__":
    
    path_to_data = 'dataset/data.txt'
    path_to_label = 'dataset/labels.xlsx'
    outdir = 'output/'

    df = data_clean(path_to_data, path_to_label)
    
    label_cols = ['Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5', 'Category_6', 'Category_7', 'Category_8', 'Category_9']

    y_score = []; y_true = []
    for i in label_cols:
        x_train, x_val, y_train, y_val = feature_encoder(df, i)
        
        xtrain_count, xvalid_count = word_embedder(x_train, x_val)
        clf, accuracy, y_pred = model(xtrain_count, y_train, xvalid_count)
        # save_pkl(i, clf, outdir)
        y_score.append(y_pred)
        y_true.append(y_val)

        print("NB, Count Vectors: ", accuracy)
    
    eval_auc(np.asarray(y_score).T, np.asarray(y_true).T)