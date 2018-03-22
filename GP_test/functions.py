#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:33:04 2017

@author: jzhao
"""

import numpy as np

def read_rdb(filename):
    
    f = open(filename, 'r')
    data = f.readlines()
    f.close()
    
    z=0
    while data[z][:2] == '# ' or data[z][:2] == ' #':
        z += 1

    key = str.split(data[z+0][:-1],'\t')
    output = {}
    for i in range(len(key)): output[key[i]] = []
    
    for line in data[z+2:]:
        qq = str.split(line[:-1],'\t')
        for i in range(len(key)):
            try: value = float(qq[i])
            except ValueError: value = qq[i]
            output[key[i]].append(value)

    return output
    
    
def gaussian(x, a, b, c, d):
    val = a * np.exp(-(x - b)**2 / c**2) + d
    return val    