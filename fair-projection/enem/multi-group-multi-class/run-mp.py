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
from utils import load_enem, MP_tol

parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--multigroup', type=bool, default=False)
parser.add_argument('--n_classes', type=int, choices=[2, 5], default=2)
parser.add_argument('--n_groups', type=int, choices=[2, 5], default=2) # 5
args = parser.parse_args()

## load ENEM dataset
enem_path = '../data/microdados_enem_2020/DADOS/' #changed to 2020
enem_file = 'MICRODADOS_ENEM_2020.csv' #changed for 2020
label = ['NU_NOTA_CH'] ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
group_attribute = ['TP_COR_RACA','TP_SEXO']
question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,25)] #changed for 2020
domestic_vars = ['SG_UF_PROVA', 'TP_FAIXA_ETARIA'] #changed for 2020
all_vars = label+group_attribute+question_vars+domestic_vars

multigroup = args.multigroup
n_classes = args.n_classes
n_groups = args.n_groups
# n_sample = 1200000
n_sample = 50000



fname = '../enem_data/enem-c'+str(n_classes) + '-g' + str(n_groups) + '-' + str(n_sample) + '-20.pkl'
if os.path.isfile(fname):
    df = pd.read_pickle(fname)
else:
    df = load_enem(enem_path, enem_file, all_vars, label, n_sample, n_classes, multigroup=multigroup)
    df.to_pickle(fname)

start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filename = 'enem-multi-'+ str(df.shape[0]) + '-' + str(n_classes) + '-' + str(n_groups) + '-' + start_time_str
f = open(filename+'-log.txt','w')

repetition = 5
use_protected = True
# tolerance = [0.001, 0.01, 0.1, 1.0]
tolerance = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

f.write('Setup Summary\n')
f.write(' Sampled Dataset Shape: ' + str(df.shape) + '\n')
f.write(' repetition: '+str(repetition) + '\n')
f.write(' use_protected: '+str(use_protected) + '\n')
f.write(' n_classes: ' + str(n_classes) + '\n')
f.write(' n_groups: ' + str(n_groups) + '\n')
f.write(' multigroup: '+str(multigroup) + '\n')
f.write(' tolerance: '+str(tolerance) + '\n')
f.flush()

## Logit
f.write('Logit - CE - meo\n')
logit_ce_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
##
f.write('Logit - KL - meo\n')
logit_kl_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')

f.write('Logit - CE - sp\n')
logit_ce_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')
##
f.write('Logit - KL - sp\n')
logit_kl_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')

## GBM
f.write('GBM - CE - meo\n')
gbm_ce_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='gbm', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
##
f.write('GBM - KL - meo\n')
gbm_kl_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')

f.write('GBM - CE - sp\n')
gbm_ce_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='gbm', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')
##
f.write('GBM - KL - sp\n')
gbm_kl_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')


## RFC
f.write('RFC - CE - meo\n')
rf_ce_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='meo')
##
f.write('RFC - KL - meo\n')
rf_kl_meo = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='rfc', div='kl', num_iter=repetition, rand_seed=42, constraint='meo')

f.write('RFC - CE - sp\n')
rf_ce_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='rfc', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint='sp')
##
f.write('RFC - KL - sp\n')
rf_kl_sp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='rfc', div='kl', num_iter=repetition, rand_seed=42, constraint='sp')

## MEO+SP
# f.write('Logit - CE - meo&sp\n')
# logit_ce_meosp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='logit', div='cross-entropy', num_iter=repetition, rand_seed=42, constraint=['meo', 'sp'])
# ##
# f.write('Logit - KL - meo&sp\n')
# logit_kl_meosp = MP_tol(df, ns=n_groups, nc=n_classes, use_protected = use_protected, tolerance=tolerance, log = f, model='logit', div='kl', num_iter=repetition, rand_seed=42, constraint=['meo', 'sp'])


save = {
    'logit_ce_meo': logit_ce_meo,
    'logit_kl_meo': logit_kl_meo,
    'logit_ce_sp': logit_ce_sp,
    'logit_kl_sp': logit_kl_sp,
    'gbm_ce_meo': gbm_ce_meo,
    'gbm_kl_meo': gbm_kl_meo,
    'gbm_ce_sp': gbm_ce_sp,
    'gbm_kl_sp': gbm_kl_sp,
    'rf_ce_meo': rf_ce_meo,
    'rf_kl_meo': rf_kl_meo,
    'rf_ce_sp': rf_ce_sp,
    'rf_kl_sp': rf_kl_sp,
    'tolerance': tolerance
}

savename = 'enem-multi-'+ str(df.shape[0]) + '-' + str(n_classes) + '-' + str(n_groups) + '-' + start_time_str + '.pkl'
with open(savename, 'wb+') as pickle_f:
    pickle.dump(save, pickle_f, 2)

f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
f.write('Finished!!!\n')
f.flush()
f.close()

