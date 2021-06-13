import os
import re
import pandas as pd
import numpy as np
import warnings
from variables import*
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle, class_weight
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore", FutureWarning)
np.random.seed(seed)

def get_csv_data(csv_path):
    '''
        Get each CSV seperately.Then filter columns and preprocess dataframe
    '''
    df = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df.columns = map(str.lower, df.columns)
    df = df[['terms', 'subtopic']]
    df = df.dropna(axis=1, how='all') 
    df['subtopic'] = df['subtopic'].str.lower()
    df['subtopic'] = df['subtopic'].str.strip()
    df = df[df['terms'].notna()]
    df = df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(method='ffill')
    return df

def load_csvs():
    '''
        Load each preprocessed dataframe from get_csv_data function and concatenate into one long dataframe
    '''
    csv_files = os.listdir(csv_file_paths)
    for i,csv_file in enumerate(csv_files):
        csv_path = os.path.join(csv_file_paths, csv_file)
        df = get_csv_data(csv_path)
        if (i == 0):
            final_df = df 
        else:
            final_df = pd.concat([final_df, df], ignore_index=True)
    final_df = final_df.dropna(axis=0, how='any') 
    final_df.to_csv(final_csv_path, index=False)

def lemmatization(lemmatizer,sentence):
    '''
        Lematize texts in the terms
    '''
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    '''
        Remove stop words in texts in the terms
    '''
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    '''
        Text preprocess on term text using above functions
    '''
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    '''
        Preprocess entire terms
    '''
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

def preprocess_labels(column): # preprocess label strings by removing digits and strips
    column = column.str.replace('[^\w\s]','')
    column = column.str.replace('[0-9]','')
    column = column.str.strip()
    column = column.str.lower()
    return column

def filter_labels(Y): # Filter labels which has less samples than min_samples
    label_count = Counter(Y)
    required_labels = [label for label, count in label_count.items() if int(count) >= min_samples]
    required_labels = np.array(required_labels)
    valid_indices = np.where(np.in1d(Y, required_labels))[0]
    return valid_indices

def load_data():
    '''
        Encode labels and then split into train and test data.
    '''
    if not os.path.exists(final_csv_path):
        load_csvs()

    df = pd.read_csv(final_csv_path)

    classes = preprocess_labels(df['subtopic']).values 
    policy_texts = df['terms'].values

    policy_texts_original = policy_texts
    policy_texts = preprocessed_data(policy_texts) # preprocess raw text

    Xoriginal, X, Y = shuffle(policy_texts_original, policy_texts, classes) # shuffle data
    valid_indices = filter_labels(Y)
    X = X[valid_indices]
    Y = Y[valid_indices]
    Xoriginal = Xoriginal[valid_indices]
    Yoriginal = Y

    encoder = LabelEncoder() # fit the label encorder with string labels
    encoder.fit(Y)
    Y = encoder.transform(Y)

    X, Y = shuffle(X, Y)
    Ntest = int(cutoff * len(Y)) # split train and test data
    rand_idxs = np.random.choice(len(Y), Ntest, replace=False)
    Xtest = X[rand_idxs,]
    Ytest = Y[rand_idxs]

    class_data = dict(Counter(Y)) # create class dict to handle class imbalance problem
    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(Y),
                                                    Y)
    class_weights = {i : class_weights[i] for i in range(len(set(Y)))}
    return X, Y, Xtest, Ytest, class_weights, encoder, Xoriginal, Yoriginal