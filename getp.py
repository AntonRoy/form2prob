import pandas as pd
import numpy as np
import string
import gensim
import pymorphy2
import pickle

w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/anton/F2P/ruscorpora.model.bin', binary=True)
with open("/home/anton/F2P/logreg.pcl", 'rb') as f:
    lg = pickle.load(f)
morph = pymorphy2.MorphAnalyzer()


def one_hot(arr):
    for i in range(len(arr)):
        langs = arr[i]
        oh_langs = [0 for i in range(13)]
        for lang in langs:
            oh_langs[lang] = 1
        arr[i] = oh_langs
    return arr


def lang(arr):
    arr = list(map(lambda x: x.lower(), arr))
    langs = []
    key_lang = {'sql':0, 'python':1,
            'питон':1, 'c++':2, 'c#':3,
            'php':4,
            'с++':2, 'с#':3, 'пхп':4,
            'r':5,
            'assembler':6, 'ассемблер':6,
                'pascal':7, 'паскаль':7,
               'java':8, 'джава': 8, 'ruby':10, 'html':10, 'css':11, 'javascript':12}
    for st in arr:
        lang = []
        for key in key_lang.keys():
            if key in st:
                lang.append(key_lang[key])
        langs.append(lang)
    return langs


def avg_feature_vector(words, num_features):
    global w2v
    model = w2v
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        nwords = nwords + 1
        try:
            featureVec = np.add(featureVec, model[word])
        except KeyError:
            featureVec = np.add(featureVec, np.zeros(num_features))

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def normalize_form(txt):
    global morph
    p = [morph.parse(x)[0].normal_form for x in txt]
    return p


def make_dict(x):
    dict_rus = ' '.join(x)
    translator = str.maketrans({key: ' ' for key in string.punctuation.replace('/', '').replace('-', '') + string.digits + '№'})
    dict_rus = dict_rus.translate(translator).lower()
    dict_rus_split = dict_rus.split()
    return dict_rus_split


def words2vecs(arr):
    arr = list(map(lambda x: x.split(), arr))
    arr = [normalize_form(make_dict(new)) for new in arr]
    arr = [avg_feature_vector(sent, 300) for sent in list(arr)]
    return arr


def predict_prob(age, langs, projects, compets):
    global lg
    all_ = pd.DataFrame()
    age = [age]
    langs = one_hot(lang([langs]))
    projects = words2vecs([projects])
    compets = words2vecs([compets])
    all_['age'] = list(age)
    projects = list(projects[0])
    compet = list(compets[0])
    langs = list(langs[0])
    for i in range(len(langs)):
        all_[str(i)] = langs[i]
    for i in range(len(projects)):
        all_['p' + str(i)] = projects[i]
    for i in range(len(compet)):
        all_['c' + str(i)] = compet[i]
    return lg.predict_proba(all_)[0][1]