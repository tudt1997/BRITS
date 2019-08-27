import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class MyTrainSet(Dataset):
    def __init__(self):
        super(MyTrainSet, self).__init__()
        self.content = open('./json/json-a').readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        rec['is_train'] = 1
        return rec


class MyValSet(Dataset):
    def __init__(self):
        super(MyValSet, self).__init__()
        self.train_content = open('./json/json-a').readlines()
        self.val_content = open('./json/json-b').readlines()
        self.test_content = open('./json/json-c').readlines()

        self.content = self.train_content + self.val_content + self.test_content
        indices = np.arange(len(self.train_content))
        self.val_indices = np.random.choice(indices, len(self.train_content) // 5)


    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = json.loads(self.content[idx])
        if idx in self.val_indices:
            rec['is_train'] = 0
        else:
            rec['is_train'] = 1
        return rec


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))

        masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: r['eval_masks'], recs)))
        forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    return ret_dict


def get_train_loader(batch_size = 64, shuffle = True):
    data_set = MyTrainSet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter


def get_val_loader(batch_size = 64, shuffle = False):
    data_set = MyValSet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
