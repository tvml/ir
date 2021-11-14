#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Tue Nov  3 16:21:50 2020
#
# @author: giorgio
# """
#%%
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%%
import sys
import re
import copy
import json
import string

import numpy as np

from ir_preprocess import preprocess_text, words_from_text

#%%
def load_indices():
    """
        Load preprocessed collection data.
        index: document-term index
        inv_index: term_document index
        titles: document keys to titles dictionary
        charmap: character to character mapping table
        n_docs: number of documents in collection
    """
    with open('inv_index.json') as f:
        [n_docs, inv_index] = json.load(f)
    with open('index.json') as f:
        [n_docs, index] = json.load(f)
    with open('titles.json') as f:
        titles = json.load(f)
    with open('charmap.json') as f:
        charmap = json.load(f)
    return index, inv_index, titles, charmap, n_docs

#%%
#
def query(q, index=index, inv_index=inv_index, n_docs=n_docs, charmap=charmap):
    """
        Return ranked list of (document key, score) in collection wrt the given query
        q: query, as list of terms
        index: document-term index
        inv_index: term_document index
        titles: document keys to titles dictionary
        charmap: character to character mapping table
        n_docs: number of documents in collection
    """
    q_text = ' '.join(q)
    t = preprocess_text(q_text, charmap)
    query_terms_list, query_terms_set = words_from_text(t)
    all_scores = get_all_scores(query_terms_list, index, inv_index, n_docs)
    sorted_list = get_sorted_result(all_scores)    
    return sorted_list#output_result(sorted_list, titles)

def output_result(results, titles=titles):
    """
        Prints a ranked list of (document title, score) from the given list of (document key, score)
    """
    if len(results.keys()) == 0:
        print('No match')
    else:
        for k, v in results.items():
            print('{0:s}: {1:5.3f}'. format(titles[k], v))

def get_sorted_result(all_scores):
    more_matches = 0
    more_matches_term = None
    for k, d in all_scores.items():
        if len(all_scores[k]) > more_matches:
            more_matches = len(all_scores[k])
            more_matches_term = k
    if more_matches_term is None:
        return {}
    result = copy.deepcopy(all_scores[more_matches_term])
    for k, docs in all_scores.items():
        if k is not more_matches_term:
            for key in docs.keys():
                if key in result.keys():
                    result[key] += docs[key]
                else:
                    result[key] = docs[key]
    sorted_result = {k: v for k, v in sorted(result.items(), key=lambda x: x[1], reverse=True)}
    return sorted_result


def get_all_scores(query_terms_list, index=index, inv_index=inv_index, n_docs=n_docs):
    """
        Computes the relevance score for each (query term, document) pair. 
        Returns a dictionary, indexed by query terms of dictionaries, indexed by document
        keys, of scores
    """
    scores = {}
    for t in query_terms_list:
        scores[t] = doc_scores(t, index, inv_index, n_docs)
    return scores

def doc_scores(t, index=index, inv_index=inv_index, n_docs=n_docs):
    """
        For a given query term, computes its relevance score wrt all documents. 
        Returns a dictionary, indexed by document keys, of scores
    """
    term_scores = {}
    if inv_index.get(t):
        for doc in inv_index[t]['tf'].keys():
            term_scores[doc] = score(t, doc, index, inv_index, n_docs)
    return term_scores

def score(t, key, index=index, inv_index=inv_index, n_docs=n_docs):
    """
        For a given pair (query term, document key), returns the corresponding
        relevance score
    """
     scr = tf(t, key, inv_index)*idf(t, inv_index, n_docs)/normalize(t, key, index)
     return scr

def tf(t, key, inv_index):
    val = inv_index[t]['tf'][key]
    return 1+np.log10(val)

def idf(t, inv_index, n_docs):
    val = inv_index[t]['df']
    return np.log10(n_docs/val)

def normalize(t, key, index):
    return index[key]['norm']

#%%
sc=get_all_scores(['frodo','bilbo'],index, inv_index, n_docs)

#%%

def main():
    index, inv_index, titles, charmap, n_docs = load_indices()

    q = ['Legolas']

    query(q, index, inv_index, titles, n_docs, charmap)

#%%
index, inv_index, titles, charmap, n_docs = load_indices()
q = ['boromir', 'legolas']
query(q, index, inv_index, titles, n_docs, charmap)


#%%


if __name__ == "__main__":
    index, inv_index, titles, charmap, n_docs = load_indices()
    q = []
    for i, arg in enumerate(sys.argv):
        q.append(arg)
    query(q[1:], index, inv_index, titles, n_docs, charmap)

#inv_index

#%%
 scores = query(q, index, inv_index, titles, n_docs, charmap)

# more_matches = 0
# more_matches_term = None
# for k, d in scores.items():
#     if len(scores[k]) > more_matches:
#         more_matches = len(scores[k])
#         more_matches_term = k

# result = copy.deepcopy(scores[more_matches_term])
# for k, docs in scores.items():
#     if k is not more_matches_term:
#         for key in docs.keys():
#             if key in result.keys():
#                 result[key] += docs[key]
#             else:
#                 result[key] = docs[key]
# sorted_result = {k: v for k, v in sorted(result.items(), key=lambda x: x[1], reverse=True)}

# for k, v in sorted_result.items():
#     print('{}: {}'. format(titles[k], v))
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.data import *
x=np.linspace(0,10,1000)
#plt.plot(x, np.exp(x))

# %%
x = np.linspace(0,10,1000)
d = {'x': x, 'exp': np.exp(x)}
df = pd.DataFrame(data=d)
(
    ggplot(df, aes(x='x', y='exp'))
    + geom_line(color='red',        # set line colour
                  size=.3,   ) # line plot
    + labs(x='$x$', y='$e^x$')
)

# len(scores['bofur'])

# t = 'théoden'
# doc = 'JRRTLetter193'
# print(t)
# print(doc)
# print(inv_index[t]['tf'][doc])
# print(inv_index[t]['df'])
# print(tf(t,doc))
# print(idf(t))
# print(normalize(doc,t))
# print(score(t,doc))

# t = 'next'
# for k,val in inv_index[t]['tf'].items():
#     print(k, val)


# f = open("archive/fr1.txt", "r")
# if f.mode == 'r':
#     text = f.read()
# lst_lines = text.splitlines( )
# line = lst_lines[0]
# g=re.match('\s*\w+\s*', line)

# %%
