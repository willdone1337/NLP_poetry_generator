import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from tensorflow.keras.models import load_model
import re
import os.path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import pandas as pd
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

import os
from google.colab import drive
drive.mount('/content/gdrive')

data = open('/content/gdrive/MyDrive/models/nizami_books.txt',encoding='utf-8').read().lower().split('\n')
data=list(data)


tokenizer=Tokenizer(num_words=3000)
tokenizer.fit_on_texts(data)
total_words=len(tokenizer.word_index)
input_sequences=[]
for line in data:
    token_list=tokenizer.texts_to_sequences([line])[0]
    for x in range(len(token_list)):
        n_gram_seq=token_list[:x+1]
        input_sequences.append(n_gram_seq)

max_lenght=max([len(x) for x in input_sequences])
padded_sequences=np.array(pad_sequences(input_sequences,maxlen=max_lenght,padding='pre'))
x,y=padded_sequences[:,:-1],padded_sequences[:,-1]
y=tf.keras.utils.to_categorical(y,num_classes=total_words)

model=keras.Sequential([
        keras.layers.Embedding(total_words,128,input_length=max_lenght),
        keras.layers.Bidirectional(LSTM(128)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(total_words/2,activation='relu'),
        keras.layers.Dense(total_words,activation='softmax')
])


model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

model.summary()

model.fit(x,y,epochs=100,verbose=1)


if os.path.isfile('/content/gdrive/MyDrive/colabv2/trynizamivol2_14may_dropout_30.h5') is False:
    model.save('/content/gdrive/MyDrive/colabv2/trynizamivol2_14may_dropout_30.h5')


seed_text = "sevirəm"
next_words = 100
import os
def poem(x):
    next_words=100
    seed_text = ''
    for _ in range(1, next_words + 1):
        token_list = tokenizer.texts_to_sequences([seed_text + x])[0]
        token_list = pad_sequences([token_list], maxlen=max_lenght - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        # if _%4==0:
        #   output_word+='\n'
        seed_text += ' ' + output_word
    return seed_text



print(poem('Ley;i'))
