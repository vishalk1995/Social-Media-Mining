# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:12:43 2019
Name:Vishal Kumar
ASU ID: 1215200480
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

ground_label = ['unrelated', 'agreed', 'disagreed']#distinct y variables
label_dict =  {'unrelated':[1,0,0],'agreed':[0,1,0],'disagreed':[0,0,1]} #For one-hot encoding

#Reverse dictionary to help convert generated confidences into appropriate labels
reverse_dict = {'[1. 0. 0.]':'unrelated','[0. 1. 0.]':'agreed','[0. 0. 1.]':'disagreed'}

#mamimum number of words from each sentence to be used
max_len = 40

#loading and preprocessing data using pandas 
df_val = pd.read_csv("Validation.csv",usecols=[5,6,7])#only reading required columns

df_val = df_val.dropna()
df_val = df_val.apply(lambda x: x.astype(str).str.lower())
df_val = df_val[df_val['label'].isin(ground_label)]
df_val = df_val.reset_index(drop=True)

#X values for neural net
X_val=df_val[['title1_en','title2_en']]
#Y values for neural net
Y_val=df_val['label']

#One-hot encoding of labels
Y_validate = []
for element in Y_val:
    Y_validate.append(label_dict[element])
Y_validate = np.array(Y_validate,dtype= 'float64')

'''tokeninzing, encoding and padding sentences
to be fed into neural net
'''
tokenv1 = Tokenizer()
tokenv1.fit_on_texts(X_val['title1_en'])
tokenv2 = Tokenizer()
tokenv2.fit_on_texts(X_val['title2_en'])
encoded_senv1 = tokenv1.texts_to_sequences(X_val['title1_en'])
encoded_senv2 = tokenv2.texts_to_sequences(X_val['title2_en'])
padded_senv1 = pad_sequences(encoded_senv1, maxlen = max_len, padding = 'post')
padded_senv2 = pad_sequences(encoded_senv2, maxlen = max_len, padding = 'post')

#loading the neural net from a saved .h5 file 
model = load_model('smm_project2_model.h5')

#evaluating and giving output against validation data
evaluate_details = model.evaluate([padded_senv1, padded_senv2], Y_validate, batch_size = 512)
print('Accuracy on validation set is : '+ str(evaluate_details[1]*100) + '%')