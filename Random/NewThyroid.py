# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:37:14 2023

@author: hp
"""

import pandas as pd


dataset = pd.read_csv("newthyroid.dat", delimiter = ',', header=None, skiprows = 10)