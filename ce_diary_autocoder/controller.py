import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from ce_diary_autocoder.a_pipeline_processing import process_data
from ce_diary_autocoder.b_vectorize import vectorize

# autocode EMLS dataset
def fit_predict_emls(data):

    #define columns and empty test set
    COLUMN_NAMES = ['MEALTYPE_1','MEALTYPE_2','MEALTYPE_3',
    'MEALTYPE_4','MEALTYPE_G','VENDOR_1','VENDOR_2','VENDOR_3',
    'VENDOR_4','VENDOR_G','TYPEALC_1','TYPEALC_12','TYPEALC_123',
    'TYPEALC_13','TYPEALC_2','TYPEALC_23','TYPEALC_3','TYPEALC_B','TYPEALC_G']

    #load pkl model
    print("load new models")
    model = joblib.load(r'models\emls_model.pkl')
    error_vectorizer = joblib.load(r'vectorizers\emls_error_vectorizer.pkl')
    error_model = joblib.load(r'models\emls_error_model.pkl')

    #split dataset into X and y
    X = data.loc[:,['MEALTYPE','VENDOR','TYPEALC']].astype(str)
    X = pd.get_dummies(X)
    #y = data['ITEM']

    #append missing columns and fill with 0
    missing_cols = set(COLUMN_NAMES) - set(X.columns)
    if len(missing_cols)>0:
        for col in missing_cols:
            X[col] = np.zeros(X.shape[0])

    #make predictions
    y_pred = model.predict(X)

    ##### Dealing with Errors #######

    #filter for errors
    wrong_pred_mask = (y_pred != data['ITEM'])
    wrong_pred = data[wrong_pred_mask]

    # rename column to itemdesc if it is currently oltname
    if 'OLTNAME' in wrong_pred.columns:
        wrong_pred = wrong_pred.rename(columns={'OLTNAME':'ITEMDESC'})

    wrong_pred['ITEMDESC'] = wrong_pred['ITEMDESC'].fillna(0)

    #vectorize errors
    data_errors = pd.DataFrame(error_vectorizer.transform(wrong_pred['ITEMDESC']).toarray(), columns=error_vectorizer.get_feature_names())

    # add mealtype and vendor features
    X_errors = X[wrong_pred_mask][COLUMN_NAMES].reset_index()
    data_errors = pd.concat([X_errors,data_errors],axis=1).drop('index',axis=1)

    # predict
    y_pred_errors = error_model.predict(data_errors)

    # replace errors with new prediction
    y_pred[wrong_pred_mask] = y_pred_errors

    # create new column for predictions
    data['Y_PRED'] = y_pred

    return data

def fit_predict(data, title):

    #load pkl model based on dataset
    if title == 'eclo':
        model = joblib.load(r'models\eclo_model.pkl')
        vectorizer = joblib.load(r'vectorizers\eclo_vectorizer.pkl')

    elif title == 'eoth':
        model = joblib.load(r'models\eoth_model.pkl')
        vectorizer = joblib.load(r'vectorizers\eoth_vectorizer.pkl')

    elif title == 'efdb':
        model = joblib.load(r'models\efdb_model.pkl')
        vectorizer = joblib.load(r'vectorizers\efdb_vectorizer.pkl')

    print("models loaded")

    # get unclassifiables probabilities and predictions
    unclassifiables_df = unclassifiables_filter(data, title, model, vectorizer)

    # send data through preprocessing pipeline
    X = process_data(data)
    print('processed')
    X, y = vectorize(X, vectorizer, title)

    # create prediction column for original dataset
    data['Y_PRED'] = model.predict(X)
    print('predicted')

    #append probabilities to predictions
    data['ITEM'] = data['ITEM'].astype('str') #make sure data type is a string
    data = pd.merge(data, unclassifiables_df, how='left')

    return data

# used in file upload and hidden div to determine the file type
def parse_contents(contents, filename):
    print("parsing contents")
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:

        global_df = pd.read_sas(io.BytesIO(decoded),format='sas7bdat')
        global_df2 = global_df.stack().str.decode('utf-8').unstack()

        if 'emls.sas7bdat' in filename:

            global_df2['COST_COM'] = global_df['COST_COM']
            global_df2['ALC_COST'] = global_df['ALC_COST']

            # get model prediction
            global_df2 = fit_predict_emls(global_df2)

            return global_df2

        else:
            global_df2['COST_COM'] = global_df['COST_COM']
            global_df2 = fit_predict(global_df2,filename[:4])

        print("global_df2 returning")
        return global_df2

    except Exception as e:
        print(e)
        # return html.Div([
        #     'There was an error processing this file.'
        # ])

    return None

def unclassifiables_filter(df, title, model, vectorizer):

    # filter for unclassifiables
    unclassifiables_df = df[(df['ITEM'].astype('int') > 899999)]

    # vectorize
    X = process_data(unclassifiables_df)
    X, y = vectorize(X, vectorizer, title)

    # predict the probability of classifying
    y_pred_prob = pd.DataFrame(model.predict_proba(X), columns=model.classes_)
    print("prob predicted")
    # create probabilities list
    columns = y_pred_prob.columns
    #get the indices for the top 3 probabilities
    sorted_indices= np.argsort(np.array(-y_pred_prob)) #sort descending
    top_three_indices = [x[:3] for x in sorted_indices]
    print("indices obtained")

    # using the above indices, get the associated class and probability
    top_three_classes = []
    top_three_probabilities = []
    for idx,row in enumerate(top_three_indices):
        top_three_classes.append([columns[x] for x in row])
        top_three_probabilities.append([round(y_pred_prob.loc[idx][columns[x]],2) for x in row])
    print("associated probabilities and classes ")
    #create dataframe using these classes and probabilities
    top_three_classes = np.array(top_three_classes)
    top_three_probabilities = np.array(top_three_probabilities)
    new_df = pd.DataFrame({'Class 1': top_three_classes[:,0],
               'Probability 1':top_three_probabilities[:,0],
               'Class 2': top_three_classes[:,1],
               'Probability 2': top_three_probabilities[:,1],
               'Class 3': top_three_classes[:,2],
               'Probability 3': top_three_probabilities[:,2]
               })

    #append to dataset
    indices = unclassifiables_df.index.values
    unclassifiables_df = unclassifiables_df.reset_index(drop=True)
    unclassifiables_df = pd.concat([unclassifiables_df,new_df],axis=1)
    unclassifiables_df.index = indices
    print("unclassifiables complete")
    return unclassifiables_df

def graph_update(data):
    #errors DataFrame
    error_df = data[(data['ITEM']!=data['Y_PRED'])]

    # split dataset into axis
    y_val = list(error_df['ITEM'].value_counts())[::-1]
    x_val = list(error_df['ITEM'].value_counts().index)[::-1]

    return x_val, y_val
