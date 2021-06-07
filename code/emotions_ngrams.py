# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:42:31 2020

@author: Noman Ashraf
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv
from cleantext import clean
import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from urduhack import normalize
from urduhack.preprocess import normalize_whitespace
from urduhack.normalization import normalize_characters
from urduhack.normalization import digits_space
from urduhack.normalization import punctuations_space
from urduhack.normalization import remove_diacritics
from urduhack.preprocess import remove_accents
from sklearn.model_selection import train_test_split
class EmotionsDataset:
    
    
    def Clean_Text(text):
        data = clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=False,                  # transliterate to closest ASCII representation
        lower=False,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # fully remove punctuation
        replace_with_url=" ",
        replace_with_email=" ",
        replace_with_phone_number=" ",
        replace_with_number=" ",
        replace_with_digit=" ",
        replace_with_currency_symbol=" ",
        lang="ur"                       # set to 'de' for German special handling
        )
        return data
    
    def RemoveEmoji(text):
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)    

    def Remove_StopWords(text):     
        stopWords = [line.strip() for line in open(r"C:\Emotion_Urdu\Code\stop_words.txt", encoding="utf8")]
        words = word_tokenize(text)
        wordsFiltered = []
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        text = ' '.join(wordsFiltered)
        return text
    
    def Normalize_Urdu(text):
        normalized_text = normalize_whitespace(text)
        normalized_text = normalize_characters(normalized_text)
        normalized_text =  normalize(normalized_text)
        normalized_text = punctuations_space(normalized_text)
        normalized_text = remove_accents(normalized_text)
        return normalized_text
    
    
    def Generate_Ngrams(self,  train_dataset_path = "" , test_data_path = "", _ngram_range=(1,1), _max_features=10, words = True, _test_size = 0.2):     

        #load train data
        df_train = pd.read_csv(train_dataset_path,usecols=['Sentences','غصہ',"نفرت","خوف","اداسی","حیرت","خوشی"], sep=',', encoding = "utf-8-sig")
        df_train = df_train.drop_duplicates(keep=False)
        
        df_train['Sentences']=df_train['Sentences'].apply(EmotionsDataset.Normalize_Urdu)
        df_train['Sentences']=df_train['Sentences'].apply(EmotionsDataset.RemoveEmoji)
        df_train['Sentences']=df_train['Sentences'].apply(EmotionsDataset.Remove_StopWords)
        
        #with open("/Users/nomanashraf/Documents/Emotions_Analysis_Urdu/Dataset/training_sentences.txt", "w") as outfile:
        #    outfile.write("\n".join(df_train.Sentences.tolist()))
        
        top_words = df_train.Sentences.str.split(expand=True).stack().value_counts()
        
        df_train['word_count'] = df_train['Sentences'].apply(lambda x: len(str(x).split()))
        print(df_train[['Sentences','word_count']].head())
        
        df_train['char_count'] = df_train['Sentences'].str.len() ## this also includes spaces
        print(df_train[['Sentences','char_count']].head())
        
        from collections import Counter
        unique_vocab = len(Counter(" ".join(df_train['Sentences'].str.lower().values.tolist()).split(" ")).items())
        
        df_stat = df_train[['word_count', 'char_count']].agg(['sum','mean'])
        
        
        print("=========== Dataset Train ===========")
        print(df_train.head())
        print(len(df_train))
        print("=========== Dataset Train ===========")
        print("\n")
        
        
        print("=========== Dataset Train Stat ===========")
        print(df_stat)
        print("Unique Vocab:" + str(unique_vocab))
        print(top_words.head(10))
        print("=========== Dataset Train Stat ===========")
        
        
        
        print("=========== Labels ===========")
        comments_labels = df_train[['غصہ',"نفرت","خوف","اداسی","حیرت","خوشی"]]
        print(comments_labels.head())
        print(comments_labels.shape)
        print("=========== Labels ===========")
        print("\n")
        
        
        test_data_path = r"C:\Emotion_Urdu\dl_code/test_ds_final.csv"
        #load test data
        df_test = pd.read_csv(test_data_path,usecols=['Sentences','غصہ',"نفرت","خوف","اداسی","حیرت","خوشی"], sep=',', encoding = "utf-8-sig")
        df_test = df_test.drop_duplicates(keep=False)
        df_test['Sentences']=df_test['Sentences'].apply(EmotionsDataset.Normalize_Urdu)
        df_test['Sentences']=df_test['Sentences'].apply(EmotionsDataset.RemoveEmoji)
        df_test['Sentences']=df_test['Sentences'].apply(EmotionsDataset.Remove_StopWords)
        
        #with open("/Users/nomanashraf/Documents/Emotions_Analysis_Urdu/Dataset/test_sentences.txt", "w") as outfile:
        #    outfile.write("\n".join(df_test.Sentences.tolist()))
        
        
        
        
        top_words_test = df_test.Sentences.str.split(expand=True).stack().value_counts()
        
        df_test['word_count'] = df_test['Sentences'].apply(lambda x: len(str(x).split()))
        print(df_test[['Sentences','word_count']].head())
        
        df_test['char_count'] = df_test['Sentences'].str.len() ## this also includes spaces
        print(df_test[['Sentences','char_count']].head())
        
        from collections import Counter
        unique_vocab_test = len(Counter(" ".join(df_test['Sentences'].str.lower().values.tolist()).split(" ")).items())
        
        df_stat_test = df_test[['word_count', 'char_count']].agg(['sum','mean'])
        
        
        print("=========== Dataset Test Stat ===========")
        print(df_stat_test)
        print("Unique Vocab:" + str(unique_vocab_test))
        print(top_words_test.head(10))
        print("=========== Dataset Test Stat ===========")
        
        
        
        print("=========== Dataset Test ===========")
        print(df_test.head())
        print(len(df_test))
        print("=========== Dataset Test ===========")
        
        print("\n")
        
  
        print("=========== Labels ===========")
        testcomments_labels = df_test[['غصہ',"نفرت","خوف","اداسی","حیرت","خوشی"]]
        print(testcomments_labels.head())
        print(testcomments_labels.shape)
        print("=========== Labels ===========")
        print("\n")


        
        xTrain = df_train['Sentences']
        yTrain = comments_labels.values
        
        xTest = df_test['Sentences']
        yTest = testcomments_labels.values
        
        print( "Training Data:" + str(xTrain.shape))
        print( "Shape of labels:" + str(yTrain.shape))
        print(yTrain[0:5])
       
        
        print( "Test Data:" + str(xTest.shape))
        print( "Shape of labels:" + str(yTest.shape))
        print(yTest[0:5])
        
        return xTrain, xTest, yTrain, yTest
        
    

