
import nltk

 import string
 import wordcloud
 import numpy as np
 import pandas as pd
 import datetime as dt
 import matplotlib.pyplot as plt
 import pylab as pl

 from collections import Counter
 from sklearn import svm
 from sklearn.pipeline import Pipeline, make_pipeline
 from sklearn.linear_model import LogisticRegression
 from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
 from sklearn.decomposition import NMF, LatentDirichletAllocation
 from sklearn.naive_bayes import MultinomialNB
 from sklearn.model_selection import train_test_split, cross_val_score
 from sklearn.model_selection import GridSearchCV
 from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

 from nltk import word_tokenize
 from nltk.corpus import stopwords
 from nltk.stem.wordnet import WordNetLemmatizer
 def parse_gz(path):
     g = gzip.open(path, 'rb')
     for l in g:
         yield eval(l)

 def convert_to_DF(path):
     i = 0
     df = {}
     for d in parse_gz(path):
         df[i] = d
         i += 1
     return pd.DataFrame.from_dict(df, orient='index')

 sports_outdoors = convert_to_DF('reviews_Baby_5.json.gz')
 import gzip
 baby= convert_to_DF('reviews_Baby_5.json.gz')
 print('Dataset size: {:,} words'.format(len(baby)))
reviews = baby['reviewText']
 stops = stopwords.words('english')
 def tokenize(text):
     tokenized = word_tokenize(text)
     no_punc = []
     for review in tokenized:
         line = "".join(char for char in review if char not in string.punctuation)
         no_punc.append(line)
     tokens = lemmatize(no_punc)
     return tokens

 def lemmatize(tokens):
     lmtzr = WordNetLemmatizer()
     lemma = [lmtzr.lemmatize(t) for t in tokens]
     return lemma

 reviews = reviews.apply(lambda x: tokenize(x))
 from nltk.corpus import sentence_polarity
 import random
 sentences = sentence_polarity.sents()
 documents = [(sent, reviews) for reviews in sentence_polarity.categories()
      for sent in sentence_polarity.sents(categories=reviews)]
 random.shuffle(documents)
 all_words_list = [word for (sent,reviews) in documents for word in sent]
 all_words = nltk.FreqDist(all_words_list)
 import nltk
 all_words = nltk.FreqDist(all_words_list)
 word_items = all_words.most_common(100)
 word_features = [word for (word, freq) in word_items]
 def document_features(document, word_features):
      document_words = set(document)
      features = {}
      for word in word_features:
          features['contains({})'.format(word)] = (word in document_words)
      return features

 featuresets = [(document_features(d,word_features), c) for (d,c) in documents]
 train_set, test_set = featuresets[50:], featuresets[:50]
 classifier = nltk.NaiveBayesClassifier.train(train_set)
 print (nltk.classify.accuracy(classifier, test_set))
0.66
 #SUBJECTIVITY
 def SL_features(document, word_features, reviews):
     document_words = set(document)
     features = {}
     for word in word_features:
         features['contains({})'.format(word)] = (word in document_words)
     # count variables for the 4 classes of subjectivity
     weakPos = 0
     strongPos = 0
     weakNeg = 0
     strongNeg = 0
     for word in document_words:
         if word in reviews:
             strength, posTag, isStemmed, polarity = reviews[word]
             if strength == 'weaksubj' and polarity == 'positive':
                 weakPos += 1
             if strength == 'strongsubj' and polarity == 'positive':
                 strongPos += 1
             if strength == 'weaksubj' and polarity == 'negative':
                 weakNeg += 1
             if strength == 'strongsubj' and polarity == 'negative':
                 strongNeg += 1
             features['positivecount'] = weakPos + (2 * strongPos)
             features['negativecount'] = weakNeg + (2 * strongNeg)
     return features

 SL_featuresets = [(SL_features(d, word_features, reviews), c) for (d,c) in documents]
 train_set, test_set = SL_featuresets[100:], SL_featuresets[:100]
 classifier = nltk.NaiveBayesClassifier.train(train_set)
 print nltk.classify.accuracy(classifier, test_set)
 print (nltk.classify.accuracy(classifier, test_set))
0.61
 train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
 classifier = nltk.NaiveBayesClassifier.train(train_set)
 print (nltk.classify.accuracy(classifier, test_set))
0.616
 classifier.show_most_informative_features(20)
 NEGATION:
for sent in list(sentences)[:50]:
    for word in sent:
      if (word.endswith("n't")):
        print(sent)

negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
 def NOT_features(document, SL_featuresets, negationwords):
     features = {}
     for word in SL_featuresets:
         features['contains({})'.format(word)] = False
         features['contains(NOT{})'.format(word)] = False
     # go through document words in order
     for i in range(0, len(document)):
         word = document[i]
         if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
             i += 1
             features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
         else:
             features['contains({})'.format(word)] = (word in word_features)
     return features

 NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
 train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
 classifier = nltk.NaiveBayesClassifier.train(train_set)
 nltk.classify.accuracy(classifier, test_set)
 classifier.show_most_informative_features(30)
 #TDIDF
tfidf_vectorizer = TfidfVectorizer(stop_words=stops)
 tfidf = tfidf_vectorizer.fit_transform(review_text)
review_text = baby["reviewText"]
 tfidf = tfidf_vectorizer.fit_transform(review_text)
 tf_vectorizer = CountVectorizer(stop_words=stops)
 tf = tf_vectorizer.fit_transform(review_text)
 tfidf_feature_names = tfidf_vectorizer.get_feature_names()
 print("Number of total features: {}".format(len(tfidf_feature_names)))
#NMF
 nmf = NMF(n_components=10, random_state=1,
           alpha=.1, l1_ratio=.5)
 lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                 learning_method='online',
                                 learning_offset=50.,
                                 random_state=0)
 num_top_words = 15
 def retrieve_top_words(model, feature_names, num_top_words):
     for idx, topic in enumerate(model.components_):
         print("Topic #{}:".format(idx), end='\n')
         print(" ".join([feature_names[i]
                         for i in topic.argsort()[:-num_top_words - 1:-1]]), end='\n\n')
     print()

 nmf_tf = nmf.fit(tf)
 nmf_ = nmf_tf.transform(tf)
 Counter([np.argmax(i) for i in nmf_])
Counter({0: 46872, 2: 25170, 8: 22100, 6: 19517, 5: 12623, 4: 9570, 1: 8948, 3: 5408, 7: 5340, 9: 5244})
 retrieve_top_words(nmf_tf, tfidf_feature_names, num_top_words)
#LDA
lda_tf = lda.fit(tf)
 lda_ = lda_tf.transform(tf)
 Counter([np.argmax(i) for i in lda_])
retrieve_top_words(lda_tf, tfidf_feature_names, num_top_words)
