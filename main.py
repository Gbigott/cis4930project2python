import pandas
import nltk
import sklearn
import gensim
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import numpy
import sys
#pd.set_option('display.max_colwidth',50)

archive_1 = pd.read_csv("test.csv")
archive_2 = pd.read_csv('train.csv')

fullData = pd.merge(archive_1,archive_2,how= 'outer')
#sybArchive2 = archive_2.sample(n = 100)

# tokenization of sentences
def Prepro_of_data(archive_gen):
    lower_archive1 = []

    for row in archive_gen['Text']:
        tokens = sent_tokenize(row)
    #save as lowercase already
        lower_archive1.append(row.lower())


    #remove digital numbers
    no_digit_archive1 = []
    for sentences in lower_archive1:
        sentences = re.sub('[0-9]','',sentences)
        no_digit_archive1.append(sentences)
    print(no_digit_archive1)
    no_contraction_archive1 = []
    for sentences in no_digit_archive1:
        sentences = re.sub(r"won\'t","will not",sentences)
        sentences = re.sub(r"can\'t", "can not", sentences)
        sentences = re.sub(r"n\'t", " not", sentences)
        sentences = re.sub(r"\'re", " are", sentences)
        sentences = re.sub(r"\'s", " is", sentences)
        sentences = re.sub(r"\'d", " would", sentences)
        sentences = re.sub(r"\'ll", " will", sentences)
        sentences = re.sub(r"\'t", " not", sentences)
        sentences = re.sub(r"\'ve", " have", sentences)
        sentences = re.sub(r"\'m", " am", sentences)
        sentences = re.sub(r"http\S+", "", sentences)
        no_contraction_archive1.append(sentences)
    #print(no_contraction_archive1)

    no_sp_archive1 = []
    for sentences in no_contraction_archive1:
        sentences = re. sub("[^a-z0-9<>]",' ', sentences)
        no_sp_archive1.append(sentences)


    # remove_w_archive1 = []
    # for sentences in no_sp_archive1:
    #     sentences = [w for w in sentences.split() if not w in stopwords.words('english')]
    #     remove_w_archive1.append(sentences)
    # #print(remove_w_archive1)

    # stemmer = SnowballStemmer("english")
    # snowball_archive1 = []
    # for sentences in no_sp_archive1:
    #     temp_sentences = []
    #     for word in sentences:
    #         stemmed_word = stemmer.stem(word)
    #         temp_sentences.append(stemmed_word)
    #     snowball_archive1.append(temp_sentences)

    # lem_archive = []
    # lemitimizer = WordNetLemmatizer()
    # Post = [[lemitimizer.lemmatize(sentences) for sentences in word_tokenize(s)] for s in no_sp_archive1]
    #
    # lem_archive.append(Post)


    #print(snowball_archive1)
    return no_sp_archive1

def bagOfWords(archive_gen):

    word2count = {}
    for data in archive_gen:
        words = word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    unique_words = list(word2count.keys())

    bag = []

    for data in archive_gen:
        words = word_tokenize(data)
        bag_vector = numpy.zeros(len(unique_words))
        for w in words:
            for i, word in enumerate(unique_words):
                if word == w:
                    bag_vector[i] += 1
        bag.append(bag_vector)
    #print(bag)

    return bag

def TFIDF(archive_gen):
    tr_idf_model = TfidfVectorizer()
    tf_idf_vector = tr_idf_model.fit_transform(archive_gen)

    tf_idf_array = tf_idf_vector.toarray()
    return tf_idf_array

def Word2(archive_gen):
    model = Word2Vec(sentences=archive_gen,vector_size=100,window=5,min_count=3, workers=4)
    model.save("word2vec.model")
    array = model.wv.vectors
    print(array)

archiv1_processed = Prepro_of_data(archive_1)
archiv2_processed = Prepro_of_data(archive_2.loc[0:1000])
TFTIDFarchive1 = TFIDF(archiv1_processed)
BOWarchive1 = bagOfWords(archiv1_processed)
TFTIDFarchive2 = TFIDF(archiv2_processed)
BOWarchive2 = bagOfWords(archiv2_processed)
#Word2(archiv1_processed)

def BOWLR(testy, trainy,testx,trainx):
    ytest = testy
    xtest = testx['Sentiment']
    ytrain = numpy.array(trainy)
    #ytrain = ytrain.reshape(ytrain.shape[1:])

    xtrain = trainx['Sentiment']




    xtrain,xval,ytrain,yval = train_test_split(xtrain,ytrain,test_size=0.3)

    lc = LogisticRegression()
    lc.fit(xtrain,ytrain)


def TFIDFLR(testy, trainy,testx,trainx):
    ytest = testy
    xtest = testx['Sentiment']
    ytrain = numpy.array(trainy)
    # ytrain = ytrain.reshape(ytrain.shape[1:])

    xtrain = trainx['Sentiment']

    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.3)

    lc = LogisticRegression()
    lc.fit(xtrain, ytrain)


def Word2LR(testy, trainy, testx, trainx):
    ytest = testy
    xtest = testx['Sentiment']
    ytrain = numpy.array(trainy)
    # ytrain = ytrain.reshape(ytrain.shape[1:])

    xtrain = trainx['Sentiment']


    xtrain,xval,ytrain,yval = train_test_split(xtrain,ytrain,test_size=0.3)

    lc = LogisticRegression()
    lc.fit(xtrain,ytrain)







BOWLR(BOWarchive1,BOWarchive2,archive_1,archive_2[0:1000])



numpy.set_printoptions(threshold =sys.maxsize)
#print(archiv1_processed)
#print(TFTIDFarchive1)
#print(BOWarchive1)









