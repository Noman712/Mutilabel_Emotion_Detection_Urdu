# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:40:06 2019

@author: Noman
"""

import warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd

from keras.preprocessing import sequence
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.utils import plot_model
from keras import backend
from keras import backend as K
from keras import models
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout, Flatten, Embedding, Bidirectional, GlobalAveragePooling1D
from keras.layers import Conv1D,Conv2D, MaxPooling2D, MaxPooling1D, GlobalMaxPooling1D

from keras.optimizers import SGD,RMSprop,Adam
from keras.regularizers import l2,l1



from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import hamming_loss


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,fbeta_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


import numpy as np
from numpy.random import seed
seed(1)

class BasicModels(object):
    
    
    
    def Shape_Reshaper(_xTrain, _xTest):    
        xTrain_arr = _xTrain
        xTest_arr = _xTest
    
        
        dim1,dim3 = _xTrain.shape
        dim2 = 1        
        
        xTrain = np.reshape(_xTrain, (dim1, dim2 , dim3)) #xTrain_arr.reshape(dim1, dim2 , dim3)
        print(xTrain.shape)       
        
        t_dim1,t_dim3 = _xTest.shape
        t_dim2 = 1
        xTest = xTest_arr.reshape(t_dim1, t_dim2 , t_dim3)
        print(xTest.shape)
        
        _input_shape = (dim2,dim3)
        print(_input_shape)
       
        
        return xTrain,xTest,_input_shape
    
    
    def Shape_Reshaper_Vec(_xTrain, _xTest): 
    
        print(_xTrain.shape)
        print(_xTest.shape)
        
        dim1, dim2, dim3 = _xTrain.shape    
        xTrain = _xTrain
        print(xTrain.shape)        
        
        t_dim1, t_dim2, t_dim3 = _xTest.shape
        xTest = _xTest
        print(xTest.shape)
        
        _input_shape = (dim2,dim3)
        print(_input_shape)

        return xTrain,xTest,_input_shape
    
    
    def Shape_Reshaper_ATT_Vec(_xTrain, _xTest): 
        
        print(_xTrain.shape)
        print(_xTest.shape)
        
        
        dim1, dim2, dim3 = _xTrain.shape    
        dim2 = 1
        xTrain = np.reshape(_xTrain, (dim1, dim2 * dim3))
        print(xTrain.shape)        
        t_dim1, t_dim2,  t_dim3 = _xTest.shape
        t_dim2 = 1
        xTest = np.reshape(_xTest, (t_dim1, t_dim2 * t_dim3))
        print(xTest.shape)
        
        dim1,dim2, dim3 = _xTrain.shape    
        
        vec_dim = dim2 * dim3
        vectors_words = dim2
        input_vec_size = dim1         
        return input_vec_size,vectors_words,vec_dim, _xTrain, _xTest
    
    def Generate_Graph(self,history):
                    # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        
    
    
  
    def CNN1D(self,_xTrain, _xTest, yTrain, yTest, n_classes = 6, _loss='binary_crossentropy', _optimizer= Adam(lr=0.0001), _metrics=['accuracy'], _epochs = 25 , _validation_split = 0.1, _batch_size = 4, _verbose = 2, graph = True, _kernel_regularizer=l2(0.001), _bias_regularizer=l2(0.001)):
        print("--------------- CNN1D ---------------")  
        
        
        if _xTrain.ndim == 2:
            xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper(_xTrain, _xTest)
        else:
            xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper_Vec(_xTrain, _xTest)
            
        model = Sequential()
        #model.add(Conv1D(16, (3), strides=3, padding='same',activation='relu', input_shape=_input_shape))
        
        model.add(Conv1D(16, (2), strides= 2, padding='same',activation='tanh', kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer, input_shape=_input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(32 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(64 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(128 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Flatten()) 
        model.add(Dense(n_classes ,activation='sigmoid'))
        model.summary()
        model.compile(loss=_loss, optimizer=_optimizer, metrics=_metrics)
        #model.compile(loss=_loss, optimizer=_optimizer, metrics=_metrics)
        
        
        history = model.fit(xTrain, yTrain, epochs=_epochs,  validation_split = _validation_split)
        
        score = model.evaluate(xTest, yTest, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        
        accuracy = model.evaluate(xTest, yTest,  verbose=_verbose)
        y_pred = model.predict_classes(xTest)
            
        print("=== Classification Report ===")
        print(classification_report(np.argmax(yTest, axis = 1),y_pred))
        print('\n')
        
        print('\n')
        print("=== Confusion Matrix ===")
        print(confusion_matrix(np.argmax(yTest, axis = 1),y_pred))
        print('\n')
        
        print("=== AUC Score ===")
        accuracy = accuracy_score(np.argmax(yTest, axis = 1), y_pred)
        print('Accuracy: %f' % accuracy)
        precision = precision_score(np.argmax(yTest, axis = 1), y_pred, average = "macro" )
        print('Precision: %f' % precision)
        recall = recall_score(np.argmax(yTest, axis = 1), y_pred, average = "macro")
        print('Recall: %f' % recall)
        f1 = f1_score(np.argmax(yTest, axis = 1), y_pred, average = "macro")
        print('F1 score: %f' % f1)
              
        
        y_true = np.array(np.argmax(yTest, axis = 1))
        
        y_pred = np.array(y_pred)
         
        
        print('Hamming loss: {0}'.format(hamming_loss(y_true, y_pred))) 
        
       
        if graph:
            self.Generate_Graph(history)
        
        print("--------------- CNN1D ---------------") 
        return model
    


    def LSTM(self,_xTrain, _xTest, yTrain, yTest, n_classes = 6, _loss='binary_crossentropy', _optimizer= Adam(lr=0.0001), _metrics=['accuracy'], _epochs = 35 , _validation_split = 0.2, _batch_size = 2, _verbose = 2, graph = False , _kernel_regularizer=l2(0.01), _bias_regularizer=l2(0.01) ):
        print("--------------- LSTM ---------------")       
        
        if _xTrain.ndim == 2:
            xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper(_xTrain, _xTest)
        else:
            xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper_Vec(_xTrain, _xTest)
        
        model = Sequential()
        #model.add(LSTM(8, input_shape=_input_shape, activation='relu' , kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer , return_sequences=True))
        model.add(LSTM(8, input_shape=_input_shape, activation='relu' , return_sequences=True))
        model.add(Dropout(0.1))
        model.add(Dense(16 ,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16 ,activation='tanh'))
        model.add(Dropout(0.3))
        #model.add(Dense(16 ,activation='relu'))
        #model.add(Dropout(0.2))
        #model.add(Dense(64 ,activation='tanh'))
        model.add(Flatten()) 
        model.add(Dense(n_classes ,activation='sigmoid'))
        model.summary()
        #model.compile(loss=_loss, optimizer= _optimizer, metrics=_metrics)
        model.compile(loss=_loss, optimizer="RMSprop", metrics=_metrics)
        history = model.fit(xTrain, yTrain, epochs=_epochs,  validation_split = _validation_split)
        
        score = model.evaluate(xTest, yTest, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        
        accuracy = model.evaluate(xTest, yTest,  verbose=_verbose)
        y_pred = model.predict_classes(xTest)

        #from sklearn.metrics import mean_squared_error
        #print("Mean Square Error:")
        #print(mean_squared_error(yTest.argmax(axis = 1), y_pred))
        
        #df = pd.DataFrame()
        #df = df.append(y_pred.tolist(), ignore_index=True)
        #df.to_csv(r"F:\Context_Abusive\predictions.csv", sep=',', encoding='utf-8', index=False)
        #print("File Saved!") 

        print("=== Classification Report LSTM ===")
        print(classification_report(yTest.argmax(axis = 1), y_pred))
        print('\n')
        print("=== Confusion Matrix LSTM ===")
        print(confusion_matrix(yTest.argmax(axis = 1), y_pred))
        print('\n')
        
        print("=== AUC Score ===")
        accuracy = accuracy_score(np.argmax(yTest, axis = 1), y_pred)
        print('Accuracy: %f' % accuracy)
        precision = precision_score(np.argmax(yTest, axis = 1), y_pred, average = "macro" )
        print('Precision: %f' % precision)
        recall = recall_score(np.argmax(yTest, axis = 1), y_pred, average = "macro")
        print('Recall: %f' % recall)
        f1 = f1_score(np.argmax(yTest, axis = 1), y_pred, average = "macro")
        print('F1 score: %f' % f1)
              
        
        y_true = np.array(np.argmax(yTest, axis = 1))
        
        y_pred = np.array(y_pred)
         
        
        print('Hamming loss: {0}'.format(hamming_loss(y_true, y_pred))) 
        
       
        if graph:
            self.Generate_Graph(history)
        
        
        return model
    
    
    
    def CNN1D_Features(self,_xTrain, _xTest, yTrain, yTest, n_classes = 6, _loss='binary_crossentropy', _optimizer= Adam(lr=0.0001), _metrics=['accuracy'], _epochs = 25, _epochs_inner = 50 , _validation_split = 0.2, _batch_size = 4, _verbose = 2, graph = True , _kernel_regularizer=l2(0.001), _bias_regularizer=l2(0.0001)):
        print("--------------- CNN1D ---------------")  
        
    
        if _xTrain.ndim == 2:
            xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper(_xTrain, _xTest)
        else:
            xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper_Vec(_xTrain, _xTest)
            
        model = Sequential()
        #model.add(Conv1D(16, (3), strides=3, padding='same',activation='relu', input_shape=_input_shape))
        
        model.add(Conv1D(16, (2), strides= 2, padding='same',activation='tanh', kernel_regularizer=_kernel_regularizer, bias_regularizer=_bias_regularizer, input_shape=_input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(32 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(64 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(128 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Flatten()) 
        model.add(Dense(n_classes ,activation='sigmoid'))
        model.summary()
        model.compile(loss=_loss, optimizer=_optimizer, metrics=_metrics)
        
        history = model.fit(xTrain, yTrain, epochs=_epochs,  validation_split = _validation_split)
        
        score = model.evaluate(xTest, yTest, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        
        if graph:
            self.Generate_Graph(history)
        
        from keras import backend as K
        output_layer = K.function([model.layers[0].input], [model.layers[2].output])
        
        #concatenate row wise
        #data = np.concatenate([xTrain,xTest], axis=0)
        #X = output_layer([data])[0]
        #print(X)
        #print(X.shape)
        #yTrain = np.concatenate([yTrain,yTest], axis=0)
        #xTrain, xTest, yTrain, yTest = train_test_split(X, yTrain, test_size = 0.2)   
        
        
        xTrain_features = output_layer([xTrain])[0]
        xTest_features = output_layer([xTest])[0]
        
        #self.LSTM(xTrain, xTest, yTrain, yTest, n_classes = 2,_epochs = _epochs_inner, _verbose=2, graph = True)
        self.LSTM(xTrain_features, xTest_features, yTrain, yTest, n_classes = 6,_epochs = _epochs_inner, _verbose=2, graph = True)
        
        #return model, layer_output
        
        return model
