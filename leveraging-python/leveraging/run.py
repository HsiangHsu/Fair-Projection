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
from DataLoader import load_data, load_hsls_imputed

parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--dataset', type=str, choices=['adult', 'compas', 'hsls', 'enem'], default='adult')
args = parser.parse_args()

start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filename = args.dataset + '-leveraging'
f = open(filename+'-log.txt','w')

## load datasets
if args.dataset == 'adult':
    df = load_data(name='adult')
    protected_attrs = ['race']
    label_name = 'income'
    model = 'gbm'
elif args.dataset == 'compas':
    df = load_data(name='compas')
    protected_attrs = ['race']
    label_name = 'is_recid'
    model = 'rfc'
elif args.dataset == 'hsls':
    hsls_path = '../hsls/data/'
    hsls_file = 'hsls_df_knn_impute_past_v2.pkl'
    df = load_hsls_imputed(hsls_path, hsls_file, [])
    protected_attrs = ['racebin']
    label_name = 'gradebin'
    model = 'rfc'
elif args.dataset == 'enem':
    enem_path = '../enem/enem_data/'
    enem_file = 'enem-50000.pkl'
    df = pd.read_pickle(enem_path+enem_file)
    protected_attrs = ['racebin']
    label_name = 'gradebin'
    model = 'logit'
else:
    f.write('Undefined Dataset')

repetition = 10
use_protected = True
use_sample_weight = True
tune_threshold = False

f.write('Setup Summary\n')
f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
f.write(' repetition: '+str(repetition) + '\n')
f.write(' use_protected: '+str(use_protected) + '\n')
f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
f.write(' tune_threshold: '+str(tune_threshold) + '\n')
f.flush()

f.write('\n')
base, leverage, _ = leveraging_approach(df, protected_attrs, label_name, use_protected, use_sample_weight, log=f, model=model, num_iter=10, rand_seed=42)

save = {
    'base': base,
    'leverage': leverage
}

savename = args.dataset + '-leveraging' +'.pkl'
with open(savename, 'wb+') as pickle_f:
    pickle.dump(save, pickle_f, 2)

f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
f.write('Finished!!!\n')
f.flush()
f.close()

