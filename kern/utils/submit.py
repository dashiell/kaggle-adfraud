#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:33:37 2018

@author: computer
"""

import pandas as pd
import numpy as np

submit_df = pd.read_csv('../input/test.csv', usecols=['click_id'])

def output_submit(filename, predictions):
    
    submit_df['is_attributed'] = predictions
    
    submit_df.to_csv('../output/' + filename, index=False)


