import pickle
import numpy as np
import pandas as pd
from flask.helpers import flash
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import preprocessor as p
import re
import string
from nltk import word_tokenize
from flask import Flask, render_template, request
from gensim import models
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM, RNN, Dense, Dropout, Embedding, RNN, Bidirectional, Add, merge, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

def clean_text(text):
    text = re.sub(r'@user','',text)
    text  = "".join([char.lower() for char in text if char not in string.punctuation])
    text  = "".join([char.lower() for char in text if char not in ['…','\n']])
    text = re.sub('[0-9]+', '', text)
    # text = ''.join(i for i in text if ord(i)<128)
    text = re.sub('\s+', ' ', text)
    text = word_tokenize(text)    
    for i in range(len(text)):
        word = text[i]
        newword = word[0]
        prev = word[0]
        prev_cnt = 0
        for j in range(1,len(word)):
            if word[j] == prev:
                prev_cnt+=1
                if prev_cnt < 2:
                    newword += word[j]
            else:
                newword += word[j]
                prev = word[j]
                prev_cnt = 0
                
        text[i] = newword

    text = [word for word in text if word not in stopword]
    #text = [str(TextBlob(word).correct()) for word in text]
    text = [word for word in text if word not in stopword]
    # text = [ps.stem(word) for word in text]
    # text = [word for word in text if 10>len(word)>2]
    # for word in text:
    #     if len(word)<=2:
    #         print(word)
    new_text = " ".join([word for word in text])
    return "tweet " + new_text


gloveEmbs = open('glove.twitter.27B/glove.twitter.27B.200d.txt', encoding='utf-8')


embeddings = {}
for line in gloveEmbs:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings[word] = coefs
gloveEmbs.close()

def embeddingOutput(X):
    """
    X: input matrix
    """
    maxLen = 10
    embDim = 200
    embOutput = np.zeros((len(X), maxLen, embDim))
    for i in range(len(X)):
        X[i] = X[i].split()
        for j in range(maxLen):
            try:
                embOutput[i][j] = embeddings[X[i][j].lower()]
            except:
                embOutput[i][j] = np.zeros((embDim, ))
    return embOutput

text = "hello world"

def addWE(text,embedding,bow_test):
    emb_text = embeddingOutput([text])
    new_text = np.concatenate((emb_text, bow_test), axis=1)
    return new_text


def get_prediction(text,mod):
    vec = pickle.load(open('./tfvec','rb'))
    p_processed = p.clean(text)
    finPre = clean_text(p_processed)
    model = pickle.load(open('./src/finalModels/modelPickles/lstmBest','rb'))
    prediction = model.predict(bow_test)

    return prediction
