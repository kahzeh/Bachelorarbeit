#!/usr/bin/python3

import sys
import jsonlines
import os
from collections import defaultdict
import re
import pickle
import chess.pgn
import chess.polyglot
import io




def tokenize(line, method = 'char'):
    tokenized_line = []
    if method == 'char':
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, 'O': 17, 'x': 18, '1/2-1/2': 18, '1-0': 20, '0-1': 21, '.': 22, '#': 23, '+':24, 'K': 25, 'R': 26, 'B': 27, 'Q': 28, 'N': 29, '-': 30}
        if '1/2-1/2' in line[-7:]:
            tokenized_line.append(token2id['1/2-1/2']) 
            for char in line[:-7]:
                if char != ' ':
                    print(char)
                    tokenized_line.append(token2id[char])
            return(tokenized_line)
        else:
            tokenized_line.append(token2id[line[-3:]])
            for char in line[:-3]:
                if char != ' ':
                    tokenized_line.append(token2id[char])
            return(tokenized_line)
    # elif method == 'wordpiece':




if len(sys.argv) != 3:
    sys.exit('Usage: ' + sys.argv[0] + 'pgn-corpus-file jsonl-file')

inputfile = sys.argv[1]
jsonlfile = sys.argv[2]
with open(inputfile, encoding='latin-1') as file:
    lines = file.readlines()
    
n = 0

games = []
game = {}

lines[:] = [line.strip() for line in lines if line.strip()]

for line in lines:

    game['id'] = n

    if line[0] == '[':
        keyvalue = line[1:-2].split(' "', 1)
        game[keyvalue[0]] = keyvalue[1]

    if line[0] == '1':           
        game['pgn-raw'] = line
        tokenized_line = tokenize(line)
        game['tokenized_pgn'] = tokenized_line
        games.append(game)
        n += 1
        game = {}
    


with jsonlines.open(jsonlfile, mode="w") as writer:  
    writer.write_all(games) 