import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

from keras.utils.vis_utils import plot_model

from keras.utils.vis_utils import plot_model

import re


import gensim

import itertools 
import speech_recognition as sr



fpath = '/trainingandtestdata/training.1600000.processed.noemoticon.csv'

cols = ['sentiment','id','date','flag','user','text']
df = pd.read_csv(fpath,header=None, names =cols, encoding = 'ISO-8859-1')

#print(df.head())

positives = df['sentiment'][df.sentiment == 4]
negatives = df['sentiment'][df.sentiment == 0]

#print('number of positive tagged sentences is:  {}'.format(len(positives)))
#print('number of negative tagged sentences is: {}'.format(len(negatives)))


def word_count(sentence):
    return len(sentence.split())
    
df['word count'] = df['text'].apply(word_count)
#df.head(3)


x = df['word count'][df.sentiment == 4]
y = df['word count'][df.sentiment == 0]
#plt.figure(figsize=(12,6))
#plt.xlim(0,45)
#plt.xlabel('word count')
#plt.ylabel('frequency')
#g = plt.hist([x, y], color=['g','r'], alpha=0.5, label=['positive','negative'])
#plt.legend(loc='upper right')

all_words = []
for line in list(df['text']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())

Counter(all_words).most_common(10)


plt.figure(figsize=(12,5))
plt.title("Top 25 most common words")
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
#fd.plot(25,cumulative=False)

word_counts = sorted(Counter(all_words).values(), reverse=True)
#plt.figure(figsize=(12,5))
#plt.loglog(word_counts, linestyle='-', linewidth=1.5)
#plt.ylabel("Freqency in dataset")
#plt.xlabel("Word Rank in frequency table")
#plt.title("log-log plot of all words")




#nltk.download('stopwords')

datacleanRE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

seqlen = 300
epsize = 8
BATCH_SIZE = 1024


POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.6)


decode_map = {'0': "NEGATIVE", '2': "NEUTRAL", '4': "POSITIVE"}
def decode_sentiment(label):
   return decode_map[str(label)]

df.sentiment = df.sentiment.apply(lambda x: decode_sentiment(x))



#print(df.head(10))




stpwords = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    text = re.sub(datacleanRE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stpwords:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x: preprocess(x))



df_train = df
test_path= '/Users/akndiwan/Desktop/Python Proj Final/trainingandtestdata/testdata.manual.2009.06.14.csv'
df_test= pd.read_csv(test_path,header=None, names =cols, encoding = 'ISO-8859-1')

df_test = df_test[df_test.sentiment != 2]

df_test.sentiment = df_test.sentiment.apply(lambda x: decode_sentiment(x))

#print(df_test.head())

#print(len(df_test))

df_test.text=df_test.text.apply(lambda x: preprocess(x))



documents = [_text.split() for _text in df_train.text] 


w2v_model = gensim.models.word2vec.Word2Vec(size=300, 
                                            window=7, 
                                            min_count=10, 
                                            workers=8)

w2v_model.build_vocab(documents)



words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
#print("Vocab size", vocab_size)

w2v_model.train(documents, total_examples=len(documents), epochs=32)



w2v_model.most_similar("silly")



tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1
#print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=seqlen)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=seqlen)



encoder = LabelEncoder()
encoder.fit(df_train.sentiment.tolist())

y_train = encoder.transform(df_train.sentiment.tolist())
y_test = encoder.transform(df_test.sentiment.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


#print("x_train", x_train.shape)
#print("y_train", y_train.shape)
#print("\n")
#print("x_test", x_test.shape)
#print("y_test", y_test.shape)


W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10


embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
#print(embedding_matrix.shape)


embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=seqlen, trainable=False)



model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

#model.summary()

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])




#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

epsize = 8
BATCH_SIZE = 2048

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epsize, validation_split=0.1, verbose=1, callbacks=callbacks)

score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
#print()
#print("Accuracy:",score[1])
#print("Loss:",score[0])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
#plt.plot(epochs, acc, 'b', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.legend()
 
#plt.figure()
 
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
 
#plt.show()


#history.history


def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def predict(text, include_neutral=True):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=seqlen)
    score = model.predict([x_test])[0]
    label = decode_sentiment(score, include_neutral=include_neutral)
    return {"label": label, "score": float(score)}

predict("Machine learning is love!")

ypred1d = []
ytest1d = list(df_test.sentiment)
scores = model.predict(x_test, verbose=1, batch_size=8000)
ypred1d = [decode_sentiment(score, include_neutral=False) for score in scores]

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)
    fm = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fm),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)




cnf_matrix = confusion_matrix(ytest1d, ypred1d)
#plt.figure(figsize=(12,12))
#plot_confusion_matrix(cnf_matrix, classes=df_train.sentiment.unique(), title="Confusion matrix")
#plt.show()


print(classification_report(ytest1d, ypred1d))

accuracy_score(ytest1d, ypred1d)


r = sr.Recognizer()
with sr.Microphone() as source:
    print("Go ahead and say something using the device microphone! \n")
    audio = r.listen(source)
    text = r.recognize_google(audio)
    print(text)

predict(text)
