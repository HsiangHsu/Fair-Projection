import pandas as pd
import numpy as np
import scipy as scp
import sklearn
import sys, os, pickle

sys.path.append(os.path.join('..'))

from FACT.helper import *
from FACT.fairness import *
from FACT.data_util import *
from FACT.plot import *
from FACT.lin_opt import *
from FACT.postprocess import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



dataname = 'enem'
model_name = 'logit'

trade_off = post_process(model_name, dataname)
result_filename = dataname+'_fact_'+model_name+'_eo_s42.pkl'
with open(result_filename, 'wb+') as f:
    pickle.dump(trade_off, f)