# need tensorflow and keras installed for this
# use GPU to make this train faster
# also need python 2.7 for this -- source activate py27
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
# from keras.preprocessing import sequence

def readData(): 
    file_name_training = 'original_training_data.xlsx'
    xl = pd.ExcelFile(file_name_training, options={'encoding':'utf-8'})
    # print(xl.sheet_names)
    df = xl.parse("training_set")

    # we expect 1785 rows of training data, but found 1783
    # get percentages
    df.loc[df['essay_set'] == 1, 'domain1_score'] *= 100/12
    # df.loc[df['essay_set'] == 8, 'domain1_score'] *= 100/60
    df.loc[df['essay_set'] == 3, 'domain1_score'] *= 100/3
    df.loc[df['essay_set'] == 4, 'domain1_score'] *= 100/3
    # convert percentages to grade cutoffs 

    # currently // workaround, 1-6 represents A-F
    df.loc[(df['domain1_score'] >= 35), 'domain1_grade'] = 5
    df.loc[(df['domain1_score'] >= 45), 'domain1_grade'] = 4
    df.loc[(df['domain1_score'] >= 55), 'domain1_grade'] = 3
    df.loc[(df['domain1_score'] >= 70), 'domain1_grade'] = 2
    df.loc[df['domain1_score'] >= 85, 'domain1_grade'] = 1
    df.loc[df['domain1_score'] < 35, 'domain1_grade'] = 6
    # preprocess and save this to csv
    
    return df.loc[(df['essay_set'] == 1) | (df['essay_set'] == 3) | (df['essay_set'] == 4)]

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

data_train = readData()
data_train = data_train.dropna(subset=['domain1_score'])
# print (data_train)
print data_train.shape

# make sure this is ascii
texts = data_train['essay'].tolist()
labels = data_train['domain1_grade'].tolist()
# labels = data_train['domain1_score'].tolist()
labels = [int(i) for i in labels]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels, dtype=str))
print (labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(7, activation='sigmoid'))
    
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)

