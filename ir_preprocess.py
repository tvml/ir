#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Nov  3 16:21:50 2020

@author: giorgio
"""
# %%
import sys
import re
import copy
import json
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('stopwords')

import numpy as np

# %%
def preprocess_text(text, charmap):
    """Preprocess text by eliminating undesired chars"""
    t = ''
    for c in list(text):
        if c in charmap.keys():
            t += charmap[c]
        else:
            t += c
    t = re.sub("'s", "", t)
    t = re.sub("(?<=[a-zA-Z])/(?=[a-zA-Z])", " ", t)
    t = re.sub("[^a-zA-Z\s]+", "", t)
    t = re.sub("\s[a-zA-Z]\s", " ", t)
    t = ' '.join([w.lower() for w in t.split() if len(w) > 1])
    return t

# %%
def words_from_text(text,
                    stop_words=set(stopwords.words('english')),
                    stemmer = PorterStemmer()):
    words = text.split()
    stop_words = stop_words
    words = [w for w in words if w not in stop_words]
    stemmer = stemmer
    stemmed = [stemmer.stem(w) for w in words]
    stemmed_set = set(stemmed)
    return stemmed, stemmed_set

# %%
def add_doc_to_indices(text, inv_ind, d1, key, charmap):
    t = preprocess_text(text, charmap)
    word_list, word_set = words_from_text(t)
    ind[key] = {}
    ind[key]['tf'] = {}
    ind[key]['norm'] = 0
    for t in word_set:
        if t in inv_ind.keys():
            inv_ind[t]['df'] += 1
        else:
            inv_ind[t] = {}
            inv_ind[t]['df'] = 1
            inv_ind[t]['tf'] = {}
        inv_ind[t]['tf'][key] = 0
    for t in word_list:
        inv_ind[t]['tf'][key] += 1
        if t in ind[key].keys():
            ind[key]['tf'][t] += 1
        else:
            ind[key]['tf'][t] = 1
    for doc in ind.keys():
        for term in ind[doc]['tf']:
            ind[doc]['norm'] += ind[doc]['tf'][term]**2
        ind[doc]['norm'] = np.sqrt(ind[doc]['norm'])

# %%
file_dict = {"fw": 0, "rs": 12, "fr": 10, "ti": 11, "tt": 10, "wr": 10,
             "rk": 9, "ho": 19, "ai": 0, "qs": 24, "ak": 0, "ta": 0,
             "JRRTLetter": 354}

# %%
def build_indices(file_dict):
    with open('charmap.json') as f:
        charmap = json.load(f)
    index = {}
    inv_index = {}
    n_docs = 0
    for k, v in file_dict.items():
        if v == 0:
            n_docs += 1
            print(k+".txt")
            f = open(k+".txt", "r")
            if f.mode == 'r':
                text = f.read()
            f.close()
            add_doc_to_indices(text, inv_index, index, k, charmap)
        else:
            for i in range(1, v+1):
                n_docs += 1
                print(k+str(i)+".txt")
                f = open(k+str(i)+".txt", "r")
                if f.mode == 'r':
                    text = f.read()
                f.close()
                add_doc_to_indices(text, inv_index, index, k+str(i), charmap)

    with open('inv_index.json', 'w') as json_file:
        json.dump([n_docs, inv_index], json_file)

    with open('index.json', 'w') as json_file:
        json.dump([n_docs, index], json_file)

    return


# %%
def main():
    build_indices()


if __name__ == "__main__":
    main()

# %%
