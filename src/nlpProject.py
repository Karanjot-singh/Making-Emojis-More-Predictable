import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from nltk.tokenize import word_tokenize
import preprocessor as p
from nltk.corpus import stopwords
from gensim import models
from tqdm import tqdm

class TfidfEmbeddingVectorizer(object):
    def __init__(self, model):
        self.model = model
        self.modelweight = None
        for i in model:
            self.dim = len(model[i])
            break
        # self.dim = len(model[model.keys()[0]])

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.modelweight = defaultdict(
		    lambda: max_idf,
		    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
		        np.mean([self.model[w] * self.modelweight[w]
		                 for w in words if w in self.model] or
		                [np.zeros(self.dim)], axis=0)
		        for words in X
		    ])

def word_embeddings(tweets, embedding):
    if embedding == "word2vec":
        X = word2vec(tweets)
        w2v = models.Word2Vec(X, vector_size=200, window=5, sg=0)
        model = dict(zip(w2v.wv.index_to_key, w2v.wv.vectors))
        
    elif embedding == "glove":
        with open("./glove.twitter.27B.200d.txt", "rb") as lines:
            model = {line.split()[0]: np.array(map(float, line.split()[1:]))
                for line in lines}


    vec = TfidfEmbeddingVectorizer(model)
    vec.fit(tweets)
    matrix = vec.transform(tweets)

    return matrix

def word2vec(tweets):
    texts = []

    for tweet in tweets:
        texts.append(tweet.split())

    return texts

train_text = pd.read_table('../dataFinal/preprocessed_train_text.txt', engine="python-fwf")
train_text = train_text['Text']
print(train_text.loc[0])

test_text = pd.read_table('../dataFinal/preprocessed_test_text.txt', engine="python-fwf")
test_text = test_text['Text']
print(test_text.loc[0])

trial_text = pd.read_table('../dataFinal/preprocessed_trial_text.txt', engine="python-fwf")
trial_text = trial_text['Text']
print(trial_text.loc[0])

train_emoji = (open("../dataFinal/finalTrainLabels.labels", "r").readlines())
for i in tqdm(range(len(train_emoji))):
    train_emoji[i] = int(train_emoji[i][0])
train_labels = pd.Series((np.array(train_emoji)).astype('int8'))
print(train_labels.loc[0])

test_emoji = (open("../dataFinal/finalTestLabels.labels", "r").readlines())
for i in tqdm(range(len(test_emoji))):
    test_emoji[i] = int(test_emoji[i][0])
test_labels = pd.Series((np.array(test_emoji)).astype('int8'))
print(test_labels.loc[0])

trial_emoji = (open("../dataFinal/finalDevLabels.labels", "r").readlines())
for i in tqdm(range(len(trial_emoji))):
    trial_emoji[i] = int(trial_emoji[i][0])
trial_labels = pd.Series((np.array(trial_emoji)).astype('int8'))
print(trial_labels.loc[0])

embedding = "word2vec"
emb_train = word_embeddings(train_text, embedding)
emb_trial = word_embeddings(trial_text, embedding)
emb_test = word_embeddings(test_text, embedding)

vec = TfidfVectorizer(min_df=1, ngram_range=(1,4), decode_error='ignore', max_features=3500)
bow_train = vec.fit_transform(train_text).toarray()
bow_trial = vec.transform(trial_text).toarray()
bow_test =  vec.transform(test_text).toarray()

train = np.concatenate((emb_train, bow_train), axis=1)
trial = np.concatenate((emb_trial, bow_trial), axis=1)
test = np.concatenate((emb_test, bow_test), axis=1)

np.save('fin_noWE_t1_train',train)
np.save('fin_noWE_t1_trial',trial)
np.save('fin_noWE_t1_test',test)