## standard packages
import sys
import os.path
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from time import localtime, strftime
import time
import argparse


## scikit learn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

## aif360
from aif360.datasets import StandardDataset

## custom packages
from utils import leveraging_approach
from DataLoader import *

parser = argparse.ArgumentParser(description = "Configuration.")
args = parser.parse_args()

start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filename = 'enem-leveraging-runtime'
f = open(filename+'-log.txt','w')

repetition = 3
enem_size = [20000, 50000, 100000, 200000, 500000, 2000000]
use_protected = True
use_sample_weight = True
tune_threshold = False
protected_attrs = ['racebin']
label_name = 'gradebin'

f.write('Setup Summary\n')
f.write(' repetition: '+str(repetition) + '\n')
f.write(' use_protected: '+str(use_protected) + '\n')
f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
f.write(' tune_threshold: '+str(tune_threshold) + '\n')
f.flush()

runtime = np.zeros((len(enem_size), repetition))

for i, size in enumerate(enem_size):
    f.write('Size: {}\n'.format(size))
    f.flush()
    ## load enem data
    fname = '../enem/enem_data/enem-' + str(size) + '.pkl'
    if os.path.isfile(fname):
        df = pd.read_pickle(fname)

    _, _, runtime[i, :] = leveraging_approach(df, protected_attrs, label_name, use_protected, use_sample_weight, log=f, model='gbm', num_iter=repetition, rand_seed=42)


save = {
    'runtime': runtime
}

savename = 'enem-leveraging-runtime' +'.pkl'
with open(savename, 'wb+') as pickle_f:
    pickle.dump(save, pickle_f, 2)

f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
f.write('Finished!!!\n')
f.flush()
f.close()

