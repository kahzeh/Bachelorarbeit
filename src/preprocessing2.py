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

    id2prop = defaultdict(lambda: defaultdict(str))
    
    n = 0

    games = []

    for line in lines:
        game = {}

        game['id'] = n

        if '[Result "*"]' in line:
            continue

        result = re.match(r'\[Result ("1\/2-1\/2"|"1-0"|"0-1")\] ', line)
        result = result.group().split()[1][1:-2]
        game['Result'] = result
        line = re.sub(r'\[Result ("1\/2-1\/2"|"1-0"|"0-1")\] ', '', line)

        whiteElo = re.match(r'\[WhiteElo "\d+"\]', line)
        whiteElo = whiteElo.group().split()[1][1:-2]
        game['WhiteElo'] = whiteElo
        line = re.sub(r'\[WhiteElo "\d+"\] ', '', line)

        blackElo = re.match(r'\[BlackElo "\d+"\]', line)
        blackElo = blackElo.group().split()[1][1:-2]
        game['BlackElo'] = blackElo
        line = re.sub(r'\[BlackElo "\d+"\] ', '', line)

        game['pgn-raw'] = line

    #    tokenizedLine = []
    #    if result == '1/2-1/2':
    #        tokenizedLine.append(token2id['1/2-1/2'])
    #        for char in line[:-9]:
    #            if char in token2id.keys():
    #                tokenizedLine.append(token2id[char])
    #    if result == '1-0':
    #        tokenizedLine.append(token2id['1-0'])
    #        for char in line[:-5]:
    #            if char in token2id.keys():
    #                tokenizedLine.append(token2id[char])
    #    if result == '0-1':
    #        tokenizedLine.append(token2id['0-1'])
    #        for char in line[:-5]:
    #            if char in token2id.keys():
    #                tokenizedLine.append(token2id[char])
    #    tokenizedLines.append(tokenizedLine)


        games.append(game)
        n += 1
    
    with open(jsonfile, "w") as outfile:  
        json.dump(games, outfile) 