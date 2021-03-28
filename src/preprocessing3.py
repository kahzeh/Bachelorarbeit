#!/usr/bin/python3

import sys
import json
import os
from collections import defaultdict
import re
import pickle
import chess.pgn
import chess.polyglot
import io


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: ' + sys.argv[0] + 'pgn-corpus-file json-file')

    inputfile = sys.argv[1]
    jsonfile = sys.argv[2]
    with open(inputfile, encoding='latin-1') as file:
        lines = file.readlines()
    
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, 'x': 16, '1/2-1/2': 17, '1-0': 18, '0-1': 19, '.': 20, '#': 21, '+':22, 'K': 23, 'R': 24, 'B': 25, 'Q': 26, 'N': 27}
    
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
            games.append(game)
            n += 1
            game = {}
        
    
    with open(jsonfile, "w") as outfile:  
        json.dump(games, outfile) 