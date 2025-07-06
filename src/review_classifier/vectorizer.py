
#%%
import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
import numpy as np
from sklearn.model_selection import train_test_split
#from ReviewVectorizer import ReviewVectorizer
from typing import Dict, List, Optional

from vocabulary import Vocabulary
from sequence_vocabulary import SequenceVocabulary


#%% Vectorizer
class NewsVectorizer(object):
    """coordinate the vocabulary and put them to use"""
    def __init__(self, title_vocab, category_vocab) -> None:
        self.title_vocab = title_vocab
        self.category_vocab = category_vocab
        
    def vectorize(self, title, vector_length=-1):
        indices = [self.title_vocab.begin_seq_index]
        indices.extend(self.title_vocab.lookup_token(token) for token in title.split(" "))
        indices.append(self.title_vocab.end_seq_index)
        
        if vector_length < 0:
            vector_length = len(indices)
            
        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.title_vocab.mask_index
        
        return out_vector
    
    @classmethod
    def from_dataframe(cls, news_df, cutoff=25):
        category_vocab = Vocabulary()
        for category in sorted(set(news_df['reviews.doRecommend'])):
            category_vocab.add_token(category)
            
        word_counts = Counter()
        for title in news_df['reviews.text']:
            for token in title.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1
                    
        title_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)
                
        return cls(title_vocab, category_vocab)
    
    @classmethod
    def from_serializable(cls, contents):
        title_vocab = SequenceVocabulary.from_serializable(contents['title_vocab'])
        
        category_vocab = Vocabulary.from_serializable(contents['category_vocab'])
        
        return cls(title_vocab=title_vocab, category_vocab=category_vocab)
    
    #@classmethod
    def to_serializable(self):
        return {'title_vocab': self.title_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}
    










