# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:21:55 2019
Name:Vishal Kumar
ASU ID: 1215200480
"""

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Embedding, merge, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import Concatenate
from keras import optimizers
from keras.utils import plot_model

ground_label = ['unrelated', 'agreed', 'disagreed'] #distinct y variables
label_dict = {'unrelated':[1,0,0],'agreed':[0,1,0],'disagreed':[0,0,1]} #For one-hot encoding
max_len = 40 #mamimum number of words from each sentence to be used

#Using pandas for data preprocessing 
df = pd.read_csv("train.csv",usecols=[5,6,7]) #only reading required columns
df = df.dropna()
df = df.apply(lambda x: x.astype(str).str.lower())
df = df[df['label'].isin(ground_label)]
X=df[['title1_en','title2_en']]
Y_value=df['label']

#One-hot encoding of labels
Y = []
for element in Y_value:
    Y.append(label_dict[element])
Y = np.array(Y,dtype= 'float64')

#Tokening the sentences into words 
token1 = Tokenizer()
token1.fit_on_texts(X['title1_en'])
token2 = Tokenizer()
token2.fit_on_texts(X['title2_en'])

vocab_size = len(token1.word_index) + len(token2.word_index) + 1

vocab_size_sen1 = len(token1.word_index) + 1
vocab_size_sen2 = len(token2.word_index) + 1

encoded_sen1 = token1.texts_to_sequences(X['title1_en'])
encoded_sen2 = token2.texts_to_sequences(X['title2_en'])


padded_sen1 = pad_sequences(encoded_sen1, maxlen = max_len, padding = 'post')
padded_sen2 = pad_sequences(encoded_sen2, maxlen = max_len, padding = 'post')

'''
using glove file to create an embedding matrix.
the file contains 400k words each having embedding of 50 dimensions.
'''
embedding_index = dict()
glove_file = open('glove.6B.50d.txt',encoding="utf8")

for line in glove_file:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embedding_index[word] = coefs
glove_file.close()

'''
Creating two seperate embedding matrices for vocabulary of sentence1 and sentence2
'''
embedding_matrix_sen1 = np.zeros((vocab_size_sen1, 50))
for word, i in token1.word_index.items():
	embedding_vector_s1 = embedding_index.get(word)
	if embedding_vector_s1 is not None:
		embedding_matrix_sen1[i] = embedding_vector_s1
        
embedding_matrix_sen2 = np.zeros((vocab_size_sen2, 50))
for word, i in token2.word_index.items():
	embedding_vector_s2 = embedding_index.get(word)
	if embedding_vector_s2 is not None:
		embedding_matrix_sen2[i] = embedding_vector_s2
        
 
'''
Building the neural network
Sending sentence 1 and 2 through two small identical neural network
and combining the their output and sending to another shallow neural network. 
'''    
    
#applying function A on sentence 1
x_sen1 = Input(shape=(max_len,))
embedding_layer_sen1 = Embedding(vocab_size_sen1, 50, weights=[embedding_matrix_sen1])(x_sen1)
embedding_layer_sen1 = Flatten()(embedding_layer_sen1)
A1_hidden = Dense(output_dim=500, activation='relu')(embedding_layer_sen1)
A1_output = Dense(output_dim = 300, activation='relu')(A1_hidden)

#applying function A on sentence 1
x_sen2 = Input(shape=(max_len,))
embedding_layer_sen2 = Embedding(vocab_size_sen2, 50, weights=[embedding_matrix_sen2])(x_sen2)
embedding_layer_sen2 = Flatten()(embedding_layer_sen2)
A2_hidden = Dense(output_dim = 500,activation='relu')(embedding_layer_sen2)
A2_output = Dense(output_dim = 300,activation='relu')(A2_hidden)

#combing the output and sending it as input for function B
concate_layer = keras.layers.concatenate([A1_output, A2_output], axis = -1)
B_hidden = Dense(output_dim = 50, activation='relu')(concate_layer)
B_output = Dense(output_dim = 3, activation='softmax')(B_hidden)

#using keras functional API to tell what is the initial input and final output.
model = Model(inputs=[x_sen1, x_sen2], outputs=B_output)

#defining out root mean square error object
rmsprop = optimizers.RMSprop(learning_rate=0.005, rho=0.9)

#compiling the model with appropriate parameters
model.compile(optimizer= rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#saving the model pictorial representation
plot_model(model, to_file='model.png')

#fitting the model on out training set
fit_details = model.fit([padded_sen1, padded_sen2], Y, epochs = 40, batch_size = 512)

#saving the model in .h5 format
model.save('smm_project2_model.h5')