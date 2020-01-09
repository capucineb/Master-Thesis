# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:46:48 2019

@author: CAPUCINE
"""
import os

def clearAll():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
        
clearAll()



def createFolder(directory):
    import os
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("FAILURE")


def get_last_file_in_path(file_path):
    from os.path import normpath, basename
    return(basename(normpath(file_path)))


