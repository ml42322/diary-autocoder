import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string

def count_punct(text):
    count = sum([1 for x in text if x in string.punctuation])
    try:
        pp = round(100*count/(len(text)-text.count(" ")),3)
    except:
        pp = 0
    return pp


def vectorize(data,vectorizer, title):
    '''
    This function preprocesses the item descriptions

    - data: past data

    '''

    #load vectorizer
    vectorizer = vectorizer

    # load column names
    if title == 'eoth':
        column_file = open('col_text/eoth_column_names.txt','r')
        columns = column_file.read().splitlines()
        columns = [x[:-1] for x in columns]

    elif title == 'efdb':
        column_file = open('col_text/efdb_column_names.txt','r')
        columns = column_file.read().splitlines()


    #transform words to vectors
    data['ITEMDESC'] = data['ITEMDESC'].fillna('')
    data['ITEM'] = data['ITEM'].astype(str)

    #train test split
    X_data = pd.DataFrame(vectorizer.transform(data['ITEMDESC']).toarray(),columns=vectorizer.get_feature_names())

    #other features that could be indicative of item code (length and punctuation)
    X_data['len']= data['ITEMDESC'].apply(lambda x: len(x)-x.count(' '))
    X_data['punct'] = data['ITEMDESC'].apply(lambda x: count_punct(x))
    X_data = X_data.fillna(0)

    y = data['ITEM']
    y = y.fillna(0)

    print("processed and vectorized")

    # reduce features for eoth and efdb
    if title == 'eoth' or title == 'efdb':
        X_data = X_data.loc[:,columns]
        X_data.fillna(0)
        print('features reduced')

    return X_data, y
