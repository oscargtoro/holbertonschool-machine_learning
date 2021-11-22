#!/usr/bin/env python3
'''
Script that creates a dictionary and creates a pd.DataFrame from that
dictionary.
'''

import pandas as pd


dic = {
    'First':{'A':0.0, 'B':0.5, 'C':1.0, 'D':1.5},
    'Second':{'A':'one', 'B':'two', 'C':'three', 'D':'four'}
    }

df = pd.DataFrame(dic)
