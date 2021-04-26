#!/usr/bin/python3
import math
import os
import re
import sys

import jsonlines
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dictionaries import charwise_dict, names_list, wordpiece_dict3


def wordpiece_tokenize(line):
    tokenizer = Tokenizer(WordPiece(wordpiece_dict3))
    tokenizer.enable_padding(length=200)
    tokenizer.enable_truncation(max_length=200)
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    output = tokenizer.encode(line)
    return(output.ids)


def tokenize(line, method = 'char'):
    tokenized_line = []
    if method == 'char':
        if '1/2-1/2' in line[-7:]:
            tokenized_line.append(charwise_dict['1/2-1/2']) 
            for char in line[:-7]:
                if char != ' ':
                    tokenized_line.append(charwise_dict[char])
            return(tokenized_line)
        else:
            tokenized_line.append(charwise_dict[line[-3:]])
            for char in line[:-3]:
                if char != ' ':
                    tokenized_line.append(charwise_dict[char])
            return(tokenized_line)

    elif method == 'wordpiece':
        if '1/2-1/2' in line[-7:]:
            tokenized_line.append(wordpiece_dict3['1/2-1/2'])
            tokenized_line += wordpiece_tokenize(line[:-7])
            return(tokenized_line)
        else:
            tokenized_line.append(wordpiece_dict3[line[-3:]])
            tokenized_line += wordpiece_tokenize(line[:-3])
            return(tokenized_line)


if len(sys.argv) != 3:
    sys.exit('Usage: ' + sys.argv[0] + ' pgn-corpus-file jsonl-file')

inputfile = sys.argv[1]
jsonlfile = sys.argv[2]
with open(inputfile, encoding='latin-1') as file:
    lines = file.readlines()
    
n = 0

games = []
game = {}

lines[:] = [line.strip() for line in lines if line.strip()]

for line in lines:

    if '*' in line[-2:]:
        continue


    game['id'] = n

    if line[0] == '[':
        keyvalue = line[1:-2].split(' "', 1)
        game[keyvalue[0]] = keyvalue[1]

    if line[0] == '1':           
        game['pgn-raw'] = line
        tokenized_line = tokenize(line, method='wordpiece')
        game['tokenized_pgn'] = tokenized_line
        games.append(game)
        n += 1
        game = {}
    
for x in games:
    if 'ECO' not in x:
        print(x['id'])
        games.remove(x)
    
    if x['White'] not in names_list:
        x['White'] = 'UNK'
    
    if x['Black'] not in names_list:
        x['Black'] = 'UNK'

    x['Date'] = x['Date'][:4]

    x['Decade'] = str(int(math.floor(int(x['Date']) / 10.0)) * 10)


with jsonlines.open(jsonlfile, mode="w") as writer:  
    writer.write_all(games) 
