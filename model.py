import re
import csv
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
logging.getLogger('tensorflow').disabled = True
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model

from variables import *
from util import*

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

np.random.seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
warnings.simplefilter("ignore", DeprecationWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class PolicyClassification:
    def __init__(self):
        X, Y, Xtest, Ytest, class_weights = load_data()
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.class_weights = class_weights
        self.size_output = len(set(self.Y))
        # self.word_cloud_visualization()
        print(" num categories : {}".format(self.size_output))
        print(" Input Shape : {}".format(self.X.shape))
        print(" Label Shape : {}".format(self.Y.shape))

    def word_cloud_visualization(self): # create word cloud to analyze most appeared words in data corpus
        policy_texts = self.X.tolist()
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
        plt.savefig('wordcloud.png')
        plt.show()

    def handle_data(self):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) # Create Tokenizer Object
        tokenizer.fit_on_texts(self.X) # Fit tokenizer with train data
        
        X_seq = tokenizer.texts_to_sequences(self.X) # tokenize train data
        self.X_pad = pad_sequences(X_seq, maxlen=max_length, truncating=trunc_type)# Pad Train data

        X_seq_test = tokenizer.texts_to_sequences(self.Xtest) # tokenize train data
        self.X_pad_test = pad_sequences(X_seq_test, maxlen=max_length)# Pad Train data
    
        self.tokenizer = tokenizer

    def feature_extractor(self): # Building the RNN model
        inputs = Input(shape=(max_length,))
        x = Embedding(output_dim=embedding_dimS, input_dim=vocab_size, input_length=max_length, name='embedding')(inputs) # Embedding layer
        x = Bidirectional(LSTM(size_lstm), name='bidirectional_lstm')(x) # Bidirectional LSTM layer
        x = Dense(dense1, activation='relu')(x)
        x = Dense(dense1, activation='relu')(x) 
        x = Dropout(keep_prob)(x)
        x = Dense(dense2, activation='relu')(x) 
        x = Dense(dense2, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        x = Dense(dense3, activation='relu')(x) 
        x = Dense(dense3, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.size_output, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        self.model = model

    def train(self): # Compile the model and training
        self.model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy']
                        )
        self.model.summary()
        self.history = self.model.fit(
                                self.X_pad,
                                self.Y,
                                validation_data = [self.X_pad_test, self.Ytest],
                                batch_size=batch_size,
                                epochs=num_epochs,
                                class_weight=self.class_weights
                                )

    def save_model(self): # Save trained model
        self.model.save(sentiment_weights)

    def load_model(self): # Load and compile pretrained model
        self.model = load_model(sentiment_weights)
        self.model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy']
                        )

    def run(self):
        if os.path.exists(sentiment_weights):
            self.load_model()
        else:
            self.handle_data()
            self.feature_extractor()
            self.train()
            self.save_model()

if __name__ == "__main__":

    if not os.path.exists(os.path.join(os.getcwd(), 'weights')):
        os.makedirs(os.path.join(os.getcwd(), 'weights'))
    model = PolicyClassification()
    model.run()
