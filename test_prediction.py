# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:02:00 2019

Name:Vishal Kumar
ASU ID: 1215200480
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#Reverse dictionary to help convert generated confidences into appropriate labels
reverse_dict = {'[1. 0. 0.]':'unrelated','[0. 1. 0.]':'agreed','[0. 0. 1.]':'disagreed'}
#mamimum number of words from each sentence to be used
max_len = 40

#loading and preprocessing data using pandas 
df_test = pd.read_csv("test.csv",usecols=[0,5,6]) #only reading required columns
df_test = df_test.dropna()
df_test = df_test.reset_index(drop=True)
df_test = df_test.apply(lambda x: x.astype(str).str.lower())

X_test=df_test[['title1_en','title2_en']]
ids = df_test['id']

'''tokeninzing, encoding and padding sentences
to be fed into neural net
'''
token_test1 = Tokenizer()
token_test1.fit_on_texts(X_test['title1_en'])
token_test2 = Tokenizer()
token_test2.fit_on_texts(X_test['title2_en'])
encoded_testsen1 = token_test1.texts_to_sequences(X_test['title1_en'])
encoded_testsen2 = token_test2.texts_to_sequences(X_test['title2_en'])
padded_testsen1 = pad_sequences(encoded_testsen1, maxlen = max_len, padding = 'post')
padded_testsen2 = pad_sequences(encoded_testsen2, maxlen = max_len, padding = 'post')

#loading the neural net from a saved .h5 file 
model = load_model('smm_project2_model.h5')

#Using model to generate output (confidence for each class)
predictions = model.predict([padded_testsen1, padded_testsen2], batch_size = 256)

#converting confidence into appropriate labels
predictions_int = np.zeros_like(predictions)
predictions_int[np.arange(len(predictions)), predictions.argmax(1)] = 1

#Generating a list of outputs
results = []
for element in predictions_int:
    results.append(reverse_dict[str(element)])

#converting above list into pandas series    
result_series = pd.Series(results)

#adding ids column 
final = pd.concat([df_test['id'], result_series], axis=1)
final.columns = ['id', 'label']

#saving file in .csv format
final.to_csv('submission.csv', sep=',', index=False)