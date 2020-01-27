from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import re


# need to install xlrd

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
    return df