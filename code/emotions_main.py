# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 06:22:35 2020

@author: Noman
"""

import Basic_Models as bm
basic_model = bm.BasicModels()

import emotions_ngrams as ed
ds_emotions = ed.EmotionsDataset()

import pandas as pd
from gensim import models
from sklearn.model_selection import train_test_split

from gensim.models.wrappers import FastText
from gensim.models.fasttext import FastText, load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors


from keras.optimizers import SGD,RMSprop,Adam
from keras.regularizers import l2,l1

import numpy
import numpy as np
from numpy.random import seed
seed(2)

#%% cell

_train_dataset_path = r"C:\Emotion_Urdu\dl_code\train_ds_final.csv"
xTrain, xTest, yTrain, yTest = ds_emotions.Generate_Ngrams(train_dataset_path = _train_dataset_path, _ngram_range=(1,1), _max_features=1000, words= True, _test_size = 0.2)


print(abc)



#%% embeding file loading 
print("=====================================")
print("word2vec file loaded")
print("=====================================")
w2v_file = "F:\Codes\\Depression_Paper_W2V_CNN\\GoogleNews-vectors-negative300.bin.gz"
w2v_file_glove = "F:\\Machine_Learning\\Basic_Models\\glove_word2vec.txt"
#w2v_file_fast_text = "C:\FasText\\cc.ur.300.bin.gz"

#w2vmodel = models.KeyedVectors.load_word2vec_format(w2v_file, binary=True,limit = 10000)
#w2vmodel = models.KeyedVectors.load_word2vec_format(w2v_file_glove)
w2v_file_fast_text = "C:\FasText\\cc.ur.300.bin.gz"
w2vmodel = FastText.load_fasttext_format(w2v_file_fast_text)   
print("Word 2 Vector File Loaded!")   
print(abc)

#%%
def Generate_w2v(w2vmodel , keywords):
    
    vector = w2vmodel['easy']
    #print( "Shape of Vector:" + str(vector.shape))
        
        
    X_train_Vector = []
    for kl in keywords:
        vector_list = []
        list_of_words = str(kl).split(" ")
        #print(list_of_words)
        for word in list_of_words:
            if word in w2vmodel.wv.vocab:
                vector_list.append(w2vmodel[word])
            else:
                vector_list.append(np.random.uniform(-0.1, 0.1, 300))
        matrix_2d = np.array(vector_list)
        average_sentence_vector = np.mean(matrix_2d, axis = 0)
        X_train_Vector.append(average_sentence_vector)
            
    X = numpy.array(X_train_Vector)
    print(X.shape)
    return X

   
 
    #%% embeding generation
print("=====================================")
print("train dataset word2vec embeddings generated")
print("=====================================")
#train_lefts, train_tweets, train_rights, train_titles
xTrain = Generate_w2v(w2vmodel, xTrain)
xTest = Generate_w2v(w2vmodel, xTest)


#%%
import Basic_Models as bm
basic_model = bm.BasicModels()
model = basic_model.CNN1D(xTrain, xTest, yTrain, yTest, n_classes = 6,  _epochs = 40, _verbose=2, graph = True)



#%%
import Basic_Models as bm
basic_model = bm.BasicModels()
model = basic_model.LSTM(xTrain, xTest, yTrain, yTest, n_classes = 6,  _epochs = 40, _verbose=2, graph = True)


#%%
import Basic_Models as bm
basic_model = bm.BasicModels()
model = basic_model.CNN1D_Features(xTrain, xTest, yTrain, yTest, n_classes = 6,  _epochs = 40, _epochs_inner = 40, _verbose=2, graph = True)

#%%
import Basic_Models as bm
basic_model = bm.BasicModels()
model = basic_model.Bi_LSTM(xTrain, xTest, yTrain, yTest, n_classes = 6,  _epochs = 5, _verbose=2, graph = True)


#%%
#model = basic_model.RF_Ngrams(xTrain1, xTest1, yTrain1, yTest1,_n_estimators = 100, _n_jobs=1, _criterion = 'entropy', _max_depth  = 15, _verbose=1)

print(abc)
#%%
model = basic_model.XGBClassifier_Ngrams(xTrain1, xTest1, yTrain1, yTest1)

#%%
model = basic_model.KNeighborsClassifier_Ngrams(xTrain1, xTest1, yTrain1, yTest1)

#%%
model = basic_model.RadiusNeighborsClassifier_Ngrams(xTrain1, xTest1, yTrain1, yTest1)

#%%

model = basic_model.MLPClassifier_Ngrams(xTrain1, xTest1, yTrain1, yTest1)

#%%
import Basic_Models as bm
basic_model = bm.BasicModels()
#model = basic_model.RF_Ngrams(xTrain1, xTest1, yTrain1, yTest1,_n_estimators = 200, _n_jobs=3, _criterion = 'entropy', _max_depth  = 15, _verbose=2)
#model = basic_model.CNN1D_Ngrams_Multicalss(xTrain1, xTest1, yTrain1, yTest1)

print(abc)
#%%




