from Data_preperation.data_prep import data_clean, label_condense
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def feature_encoder(df):

    x_train, x_validation, y_train, y_validation = model_selection.train_test_split( \
                                                        df['Privacy_Policies'], \
                                                        df['labels'])
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_validation = encoder.fit_transform(y_validation)

    return x_train, x_validation, y_train, y_validation

def model(x_train, x_val, y_train, y_val):
    # Define a pipeline combining a text feature extractor with multi lable classifier
    NB_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                    ('clf', OneVsRestClassifier(MultinomialNB(
                        fit_prior=True, class_prior=None))),
                ])
    for category in categories:
        print('... Processing {}'.format(category))
        # train the model using X_dtm & y
        NB_pipeline.fit(x_train, train[category])
        # compute the testing accuracy
        prediction = NB_pipeline.predict(x_val)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

if __name__ == "__main__":
    
    path_to_data = 'dataset/data.txt'
    path_to_label = 'dataset/labels.xlsx'

    df = data_clean(path_to_data, path_to_label)
    label_condense(df)
    # x_train, x_val, y_train, y_val = feature_encoder(df)
    # model(x_train, x_val, y_train, y_val)
    
