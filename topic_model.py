
import re
import csv
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

from variables import *
from util import*

np.random.seed(seed)
warnings.simplefilter("ignore", DeprecationWarning)

class PolicyTopics:
    def __init__(self):
        Xtrain, Xtest, Ytrain, Ytest = load_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest
        self.size_output = len(set(self.Ytest))
        self.word_cloud_visualization()

    def word_cloud_visualization(self):
        policy_texts = self.Xtrain.tolist() + self.Xtest.tolist()
        long_string = ','.join(list(policy_texts))
        wordcloud = WordCloud(
                            background_color="white", 
                            max_words=vocab_size, 
                            contour_width=3, 
                            contour_color='steelblue'
                            )
        wordcloud.generate(long_string)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def fit_embedding(self):
        vectorizer = CountVectorizer(                     
                                stop_words='english',             
                                lowercase=True,                   
                                token_pattern='[a-zA-Z0-9]{3,}', 
                                max_features=vocab_size,             
                                )
        vectorizer.fit(self.Xtrain)
        train_embeddings = vectorizer.transform(self.Xtrain)
        test_embeddings = vectorizer.transform(self.Xtest)
        self.train_embeddings = train_embeddings
        self.test_embeddings = test_embeddings
        self.vectorizer = vectorizer

    def check_sparsicity(self):
        train_dense = self.train_embeddings.todense()
        print("Train sparsicity: ", (
                                    (train_dense > 0).sum()/train_dense.size)*100, 
                                    "%"
                                    )

        test_dense = self.test_embeddings.todense()
        print("Test sparsicity: ", (
                                   (test_dense > 0).sum()/test_dense.size)*100, "%"
                                   )

    def train(self):
        lda = LDA(
                n_components=n_topics,            
                max_iter=max_iter,               
                learning_method='online',   
                random_state=seed,         
                batch_size=batch_size,          
                evaluate_every = -1,      
                n_jobs = -1,             
                )
        self.lda = lda.fit(self.train_embeddings)
        self.Ptrain = self.lda.transform(self.train_embeddings)
        self.Ptest = self.lda.transform(self.test_embeddings)
        print(self.test_embeddings.shape)
        print(self.Ptest.shape)

    def evaluation(self, lda):
        print("\nFor train data:")
        print("     Log Likelihood: ", lda.score(self.train_embeddings))
        print("     Perplexity: ", lda.perplexity(self.train_embeddings))

        print("For test data:")
        print("     Log Likelihood: ", lda.score(self.test_embeddings))
        print("     Perplexity: ", lda.perplexity(self.test_embeddings))
        
    def grid_search(self):
        lda = LDA()
        self.model = GridSearchCV(lda, param_grid=search_params)
        self.model.fit(self.train_embeddings)

    def best_model(self):
        best_lda_model = self.model.best_estimator_
        print("Best Model Params: ", self.model.best_params_)
        self.evaluation(best_lda_model)
        # {'learning_decay': 0.5, 'n_components': 10}
        
    def run(self):
        self.fit_embedding()
        self.check_sparsicity() 
        self.train()
        # self.grid_search()
        # self.best_model()

if __name__ == "__main__":
    model = PolicyTopics()
    model.run()
