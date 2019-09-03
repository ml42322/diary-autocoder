import numpy as np
import pandas as pd
import re
import json

def process_data(data):

    #load spellchecker to a dictionary
    with open('json/spellchecker2.json', 'r') as f:
        spellchecker = json.load(f)

    # right now only deals with ECLO
    COLUMN_NAMES = ['ITEM','ITEMDESC']
    
    data = data.loc[:,COLUMN_NAMES]

    # remove nonalphabetic characters
    data['ITEMDESC']=data.ITEMDESC.apply(''.join).str.replace('[^A-Za-z]+', ' ')

    #fix spelling errors
    descriptions_arr = [x.split(' ') for x in data['ITEMDESC']]
    for idx, row in enumerate(descriptions_arr):
        desc_to_check = row

        for word in row:
            try:
                if word.lower() == spellchecker[word.lower()]:
                    continue
                correction = spellchecker[word.lower()]
                desc_to_check.append(correction.upper())

            except:
                continue
        descriptions_arr[idx] = desc_to_check

    restring_desc = [' '.join(x) for x in descriptions_arr]

    # replace itemdesc column
    data['ITEMDESC'] = restring_desc

    print('Data Cleaning Complete')

    return data
