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
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ## comment out if using GPU

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
from utils import load_enem20, MP_tol

parser = argparse.ArgumentParser(description = "Configuration.")
parser.add_argument('--multigroup', type=bool, default=False)
parser.add_argument('--n_classes', type=int, choices=[2, 5], default=2)
parser.add_argument('--n_groups', type=int, choices=[2, 5], default=2) # 5
args = parser.parse_args()

start_time = time.localtime()
start_time_str = strftime("%Y-%m-%d-%H.%M.%S", start_time)
filename = 'enem-runtime-' + start_time_str
f = open(filename+'-log.txt','w')

## load ENEM dataset
repetition = 2
use_protected = True
use_sample_weight = True
tune_threshold = False
# tolerance = [0.001, 0.005, 0.01]
tolerance = [0.01]
enem_size = [20000, 50000, 100000, 200000, 500000, 1000000, 1400000]
# enem_size = [200000, 500000, 1400000]
# enem_size = [1400000]

f.write('Setup Summary\n')
f.write(' repetition: '+str(repetition) + '\n')
f.write(' use_protected: '+str(use_protected) + '\n')
f.write(' use_sample_weight: '+str(use_sample_weight) + '\n')
f.write(' tune_threshold: '+str(tune_threshold) + '\n')
f.write(' tolerance: '+str(tolerance) + '\n')
f.write(' enem size: '+str(enem_size) + '\n')
f.flush()


## ENEM-2020
enem_path = '../data/microdados_enem_2020/DADOS/' #changed to 2020
enem_file = 'MICRODADOS_ENEM_2020.csv' #changed for 2020
label = ['NU_NOTA_CH'] ## Labels could be: NU_NOTA_CH=human science, NU_NOTA_LC=languages&codes, NU_NOTA_MT=math, NU_NOTA_CN=natural science
group_attribute = ['TP_COR_RACA','TP_SEXO']
question_vars = ['Q00'+str(x) if x<10 else 'Q0' + str(x) for x in range(1,25)] #changed for 2020
domestic_vars = ['SG_UF_PROVA', 'TP_FAIXA_ETARIA'] #changed for 2020
all_vars = label+group_attribute+question_vars+domestic_vars
n_classes = 2

data_size = np.zeros((len(enem_size)))
meo_np, meo_tf = {}, {}
sp_np, sp_tf = {}, {}
for i, size in enumerate(enem_size):
    f.write('Size: {}\n'.format(size))
    f.flush()
    ## load enem data
    fname = '../enem_data/enem-' + str(size) + '.pkl'
    if os.path.isfile(fname):
        df = pd.read_pickle(fname)
    else:
        df = load_enem20(enem_path, enem_file, all_vars, label, size, n_classes, multigroup=False)
        df.to_pickle(fname)

    sp_tf[i] = MP_tol(df, use_protected=use_protected, use_sample_weight=use_sample_weight, tune_threshold=tune_threshold, 
                    tolerance=tolerance, log=f, model='gbm', div='kl', num_iter=repetition, rand_seed=42, constraint='sp', projection_method='tf')

                                                                                                                  tune_threshold=tune_threshold, tolerance=tolerance, log=f, model='gbm', div='kl',
    #                                                                                                                       num_iter=repetition, rand_seed=42, constraint='sp')


    data_size[i] = df.shape[0]

save = {
    'sp_tf': sp_tf,
    'data_size': data_size
}

savename = 'enem-runtime-' + '-' +start_time_str+'.pkl'
with open(savename, 'wb+') as pickle_f:
    pickle.dump(save, pickle_f, 2)

f.write('Total Run Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(start_time))/60))
f.write('Finished!!!\n')
f.flush()
f.close()

