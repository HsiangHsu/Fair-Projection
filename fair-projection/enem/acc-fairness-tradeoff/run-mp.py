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
from utils import load_enem, MP_tol

## load ENEM dataset
# enem_path = '../data/Microdados Enem 2017/DADOS/'
# enem_file = 'MICRODADOS_ENEM_2017.csv'
enem_path = '../data/microdados_enem_2020/DADOS/' #changed to 2020
enem_file = 'MICRODADOS_ENEM_2020.csv' #changed for 2020
label = ['NU_NOTA_CH'] ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
group_attribute = ['TP_COR_RACA','TP_SEXO']
question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,25)] #changed for 2020
domestic_vars = ['SG_UF_PROVA', 'TP_FAIXA_ETARIA'] #changed for 2020
all_vars = label+group_attribute+question_vars+domestic_vars

# question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,28)]
# domestic_vars = ['SG_UF_RESIDENCIA', 'NU_IDADE']
# article_vars = ['NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
# all_vars = label+group_attribute+question_vars+domestic_vars
n_sample = 50000
n_classes = 2

fname = '../enem_data/enem-'+str(n_sample)+'-20.pkl'
if os.path.isfile(fname):
    df = pd.read_pickle(fname)
else:
    # df = load_enem(enem_path, enem_file, all_vars, label, n_sample)
    df = load_enem(enem_path, enem_file, all_vars, label, n_sample, n_classes, multigroup=False)
    df.to_pickle(fname)

df['gradebin'] = df['gradebin'].astype(int)

start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filename = 'enem-'+ str(df.shape[0]) +'-mp-' + start_time_str
f = open(filename+'-log.txt','w')

repetition = 10
use_protected = True
use_sample_weight = True
tune_threshold = False
# tolerance = [0.000, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
tolerance = [0.000, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

f.write('Setup Summary\n')
f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
f.write(' repetition: '+str(repetition) + '\n')
f.write(' use_protected: '+str(use_protected) + '\n')
f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
f.write(' tune_threshold: '+str(tune_threshold) + '\n')
f.write(' tolerance: '+str(tolerance) + '\n')
f.flush()

### CE
## GBM
f.write('GMB - CE - meo\n')
gbm_ce_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
# ##
f.write('GMB - CE - sp\n')
gbm_ce_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')
# # ## Logit
f.write('Logit - CE - meo\n')
logit_ce_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
# ##
f.write('Logit - CE - sp\n')
logit_ce_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')
## Random Forest
f.write('RFC - CE - meo\n')
rfc_ce_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
##
f.write('RFC - CE - sp\n')
rfc_ce_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')



## KL
# GBM
f.write('GMB - KL - meo\n')
gbm_kl_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')
#
f.write('GMB - KL - sp\n')
gbm_kl_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')
# Logit
f.write('Logit - KL - meo\n')
logit_kl_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')
#
f.write('Logit - KL - sp\n')
logit_kl_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')
# Random Forest
f.write('RFC - KL - meo\n')
rfc_kl_meo = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')
#
f.write('RFC - KL - sp\n')
rfc_kl_sp = MP_tol(df, use_protected = use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, tolerance=tolerance, log = f, model='rfc', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')


save = {
    'gbm_ce_meo': gbm_ce_meo,
    'gbm_ce_sp': gbm_ce_sp,
    'logit_ce_meo': logit_ce_meo,
    'logit_ce_sp': logit_ce_sp,
    'rfc_ce_meo': rfc_ce_meo,
    'rfc_ce_sp': rfc_ce_sp,
    'gbm_kl_meo': gbm_kl_meo,
    'gbm_kl_sp': gbm_kl_sp,
    'logit_kl_meo': logit_kl_meo,
    'logit_kl_sp': logit_kl_sp,
    'rfc_kl_meo': rfc_kl_meo,
    'rfc_kl_sp': rfc_kl_sp,
    'tolerance': tolerance
}

savename = 'enem-mp-'+start_time_str+'.pkl'
with open(savename, 'wb+') as pickle_f:
    pickle.dump(save, pickle_f, 2)

f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
f.write('Finished!!!\n')
f.flush()
f.close()

