# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:28:44 2022

@author: 16028
"""

import numpy as np
import pandas as pd

def reshape_data(x_dim, y_dim, X_trainer, X_tester):
    shape_tuple = (x_dim, y_dim)
    X_trainer_reshape = []
    for i in range(X_trainer.shape[0]):
        X_trainer_reshape.append(X_trainer[i].reshape(shape_tuple))
    X_trainer_reshape = np.stack(X_trainer_reshape, axis=0 )
    print(X_trainer_reshape.shape)

    X_tester_reshape = []
    for i in range(X_tester.shape[0]):
        X_tester_reshape.append(X_tester[i].reshape(shape_tuple))
    X_tester_reshape = np.stack(X_tester_reshape, axis=0 )
    print(X_tester_reshape.shape)
    
    return X_trainer_reshape, X_tester_reshape
    
def floatify_data(X_trainer, X_tester, y_trainer, y_tester): #NOTE: I WAS USING FLOAT32 BEFORE AT SOME POINT--WHY?
    X_trainer = X_trainer.astype(np.float64)
    X_tester = X_tester.astype(np.float64)
    y_trainer = y_trainer.astype(np.float64)
    y_tester = y_tester.astype(np.float64)
    return X_trainer, X_tester, y_trainer, y_tester