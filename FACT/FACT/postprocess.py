import pandas as pd
import numpy as np
import sys, os, pickle

sys.path.append(os.path.join('..'))

from FACT.helper import *
from FACT.fairness import *
from FACT.data_util import *
from FACT.plot import *
from FACT.lin_opt import *

import time

def post_process(model_name, dataname ):
    X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, dtypes, dtypes_, sens_idc, race_idx, sex_idx = get_dataset(dataname)

    start = time.time()
    
    if model_name == 'logit':
        model_names = ['LogisticRegression']
    elif model_name == 'rf':
        model_names = ['RandomForest']
    elif model_name == 'gbm':
        model_names = ['GradientBoost']
    result = create_and_train_models(model_names, dtypes_, X_train_removed, y_train, 
                                 X_test=X_test_removed, y_test=y_test, data_name=dataname)

    grid = result[0]['estimator']
    filename = dataname + '-' + model_name + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(grid, f)

    fm = FairnessMeasures(X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, 
                        grid, sens_idc, pos_label=1, neg_label=0)

    _, mats_dict, M_const, b_const = get_fairness_mats(fm)

    # Post process
    A, b = mats_dict['EqOdd']
    A_const = M_const

    eps_vals = np.logspace(0, -6, 100)
    eps_used = []
    acc_vals = []
    for eps in eps_vals:
        try:
            res, _ = model_spec_solve(A, b, A_const, b_const, fm, eps=eps)
            acc_vals.append(1 - res.value)
            eps_used.append(eps)
        except:
            continue

    # acc_target = 0.65

    # eps_targets = np.array(eps_used)[np.array(acc_vals) > acc_target]
    # print(eps_targets)

    ## Hard-coding optimal epsilon values 
    if dataname == 'adult' and model_name == 'rf':
        eps_targets =[eps_used[41]]
    elif dataname == 'adult' and model_name == 'gbm':
        eps_targets =[eps_used[38]]
    elif dataname == 'adult' and model_name == 'logit':
        eps_targets =[eps_used[35]]
    else:
        eps_targets=[1]
    # eps_mean, acc_mean = [], []

    for eps in eps_targets:
        fact_sol_eps,  fact_sol_acc, fact_time = [], [], []

        print(eps)
        res, sol = model_spec_solve(A, b, A_const, b_const, fm, eps=eps)
        if res.status == 'infeasible':
            print('early quit: %f'%eps)
            break
        #print(eps, 1- res.value)
        zz = sol.value
        # get mixing rates

        # base classifier
        fpr_base_pos = fm.pos_group_stats['FPR']
        tpr_base_pos = fm.pos_group_stats['TPR']
        fnr_base_pos = fm.pos_group_stats['FNR']
        tnr_base_pos = fm.pos_group_stats['TNR']
        fpr_base_neg = fm.neg_group_stats['FPR']
        tpr_base_neg = fm.neg_group_stats['TPR']
        fnr_base_neg = fm.neg_group_stats['FNR']
        tnr_base_neg = fm.neg_group_stats['TNR']

        # derived classifier
        fpr_der_pos = zz[2] / (zz[2] + zz[3])
        tpr_der_pos = zz[0] / (zz[0] + zz[1])
        fnr_der_pos = 1 - tpr_der_pos
        tnr_der_pos = 1 - fpr_der_pos
        fpr_der_neg = zz[6] / (zz[6] + zz[7])
        tpr_der_neg = zz[4] / (zz[4] + zz[5])
        fnr_der_neg = 1 - tpr_der_neg
        tnr_der_neg = 1 - fpr_der_neg
        
        for i in range(10):

            # get mixing rates
            mix_mat_pos = np.array([[tpr_base_pos, fnr_base_pos, 0, 0], [fpr_base_pos, tnr_base_pos, 0, 0], [0, 0, tpr_base_pos, fnr_base_pos], [0, 0, fpr_base_pos, tnr_base_pos]])
            mix_b_pos = np.array([tpr_der_pos, fpr_der_pos, fnr_der_pos, tnr_der_pos])
            mix_sol_pos = np.linalg.solve(mix_mat_pos, mix_b_pos)
            mix_mat_neg = np.array([[tpr_base_neg, fnr_base_neg, 0, 0], [fpr_base_neg, tnr_base_neg, 0, 0], [0, 0, tpr_base_neg, fnr_base_neg], [0, 0, fpr_base_neg, tnr_base_neg]])
            mix_b_neg = np.array([tpr_der_neg, fpr_der_neg, fnr_der_neg, tnr_der_neg])
            mix_sol_neg = np.linalg.solve(mix_mat_neg, mix_b_neg)
            p2p_pos, n2p_pos, p2n_pos, n2n_pos = mix_sol_pos
            p2p_neg, n2p_neg, p2n_neg, n2n_neg = mix_sol_neg

            # Post-process accordingly
            pos_pred = fm.y_pred[fm.pos_group]
            pos_pred_fair = pos_pred.copy()
            pos_pp_indices, = np.nonzero(pos_pred)
            pos_pn_indices, = np.nonzero(1 - pos_pred)
            np.random.seed(i)
            np.random.shuffle(pos_pp_indices)
            np.random.seed(i)
            np.random.shuffle(pos_pn_indices)
            n2p_indices = pos_pn_indices[:int(len(pos_pn_indices) * n2p_pos)]
            pos_pred_fair[n2p_indices] = 1 - pos_pred_fair[n2p_indices]
            p2n_indices = pos_pp_indices[:int(len(pos_pp_indices) * (1 - p2p_pos))]
            pos_pred_fair[p2n_indices] = 1 - pos_pred_fair[p2n_indices]

            neg_pred = fm.y_pred[fm.neg_group]
            neg_pred_fair = neg_pred.copy()
            neg_pp_indices, = np.nonzero(neg_pred)
            neg_pn_indices, = np.nonzero(1 - neg_pred)
            np.random.shuffle(neg_pp_indices)
            np.random.shuffle(neg_pn_indices)
            n2p_indices = neg_pn_indices[:int(len(neg_pn_indices) * n2p_neg)]
            neg_pred_fair[n2p_indices] = 1 - neg_pred_fair[n2p_indices]
            p2n_indices = neg_pp_indices[:int(len(neg_pp_indices) * (1 - p2p_neg))]
            neg_pred_fair[p2n_indices] = 1 - neg_pred_fair[p2n_indices]

            # Measure performance

            # EOd gap and accuracy
            fn_pos = np.where(pos_pred_fair - fm.y_test[fm.pos_group] == -1)[0].shape[0]
            fp_pos = np.where(pos_pred_fair - fm.y_test[fm.pos_group] == 1)[0].shape[0]
            eod_pos = (fn_pos / fm.pos_group_num, fp_pos / fm.pos_group_num)

            fn_neg = np.where(neg_pred_fair - fm.y_test[fm.neg_group] == -1)[0].shape[0]
            fp_neg = np.where(neg_pred_fair - fm.y_test[fm.neg_group] == 1)[0].shape[0]
            eod_neg = (fn_neg / fm.neg_group_num, fp_neg / fm.neg_group_num)

            eod_gap = np.abs(eod_pos[0] - eod_neg[0]) + np.abs(eod_pos[1] - eod_neg[1])
            meo = eod_gap/2
            acc = 1 - (fn_pos + fp_pos + fn_neg + fp_neg) / fm.y_test.shape[0] 
            
            end = time.time()
            
            print(eod_gap, acc)
            
            fact_sol_eps.append(meo)
            fact_sol_acc.append(acc)
            fact_time.append(end-start)
            
            
        # eps_mean.append(fact_sol_eps)
        # acc_mean.append(fact_sol_acc)
        
    return {'meo': fact_sol_eps, 'acc': fact_sol_acc, 'time': fact_time}
