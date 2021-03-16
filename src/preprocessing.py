#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import json
import os
from collections import defaultdict

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: ' + sys.argv[0] + 'pgn-corpus-file output-file json-file')

    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    jsonfile = sys.argv[3]
    with open(inputfile) as file:
        lines = file.readlines()
    
    output = ''
    id2prop = defaultdict(lambda: defaultdict(str))
    n = 0
    for line in lines:
        if line[0] == '[':
            line = line[1:-2].split()
            prop = line[0]
            value = line[1].replace('"','')
            id2prop[n][prop] = value
            if prop == 'Result':
                output += value + ' '
        elif line[0].isdigit():
            line_updated = line.replace('\n', '')
            output += line_updated + ' '
            if ((line[-8:-1].find('1-0') != -1) or (line[-8:-1].find('0-1') != -1)):
                output = output[:-4]
                output += '\n'
                n += 1
            elif (line[-8:-1].find('1/2-1/2') != -1):
                output = output[:-8]
                output += '\n'
                n += 1

    with open(outputfile, 'w') as x:
        x.write(output)
    
    with open(jsonfile, "w") as outfile:  
        json.dump(id2prop, outfile) 
