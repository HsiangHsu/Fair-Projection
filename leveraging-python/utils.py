## standard packages
import sys
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from time import localtime, strftime
import time

## scikitlearn
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss, roc_auc_score, f1_score

## aif360
from aif360.datasets import StandardDataset

def get_idx_wo_protected(feature_names, protected_attrs):
    idx_wo_protected = set(range(len(feature_names)))
    protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
    idx_wo_protected = list(idx_wo_protected - set(protected_attr_idx))
    return idx_wo_protected

def get_idx_w_protected(feature_names):
    return list(set(range(len(feature_names))))

def get_idx_protected(feature_names, protected_attrs):
    protected_attr_idx = [feature_names.index(x) for x in protected_attrs]
    idx_protected = list(set(protected_attr_idx))
    return idx_protected

def confusion(y, y_pred):
    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

def statistical_parity_difference(y, s):
    sp0 = y[s==0].mean()
    sp1 = y[s==1].mean()
    return np.abs(sp1-sp0)

def odd_diffs(y, y_pred, s):
    y0, y1 = y[s==0], y[s==1]
    y_pred0, y_pred1 = y_pred[s==0], y_pred[s==1]

    tpr0, fpr0 = confusion(y0, y_pred0)
    tpr1, fpr1 = confusion(y1, y_pred1)

    tpr_diff = tpr1 - tpr0
    fpr_diff = fpr1 - fpr0

    return np.abs((tpr_diff + fpr_diff)) / 2

def search_threshold(clf, X, y, s):
    thresholds = np.arange(0.0, 1.0, 0.01)
    acc_score = np.zeros((len(thresholds)))
    y_prob = np.squeeze(clf.predict_proba(X=X, s=s), axis=2)

    for i, t in enumerate(thresholds):
        # Corrected probabilities
        y_pred = (y_prob[:, 1] > t).astype('int')
        # Calculate the acc scores
        acc_score[i] = accuracy_score(y, y_pred)

    index = np.argmax(acc_score)
    thresholdOpt = thresholds[index]
    return thresholdOpt

# def evaluation(idx1, idx2, clf, X, y, s, t,
#                acc, brier, auc, meo, meo_abs, mo, sp):
#     y_prob = np.squeeze(clf.predict_proba(X=X, s=s), axis=2)
#     y_pred = (y_prob[:, 1] > t).astype('int')
#
#     acc[idx1, idx2] = accuracy_score(y, y_pred)
#     brier[idx1, idx2] = brier_score_loss(y, y_prob[:, 1])
#     auc[idx1, idx2] = roc_auc_score(y_true=y, y_score=y_prob[:, 1])
#     meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs(y, y_pred, s)
#     sp[idx1, idx2] = statistical_parity_difference(y_pred, s)
#     return

def evaluate_leverage(X, y, y_pred, s):
    ## compute difference of equal opportunity (DEO)
    ## DEO
    err = (y_pred != y)
    err_0 = err[(s == 0) & (y == 1)]
    err_1 = err[(s == 1) & (y == 1)]
    deo = np.abs(err_0.mean()-err_1.mean())

    ## other metrics
    acc = accuracy_score(y, y_pred)
    meo = odd_diffs(y, y_pred, s)
    sp = statistical_parity_difference(y_pred, s)
    return acc, meo, deo, sp

def leveraging_approach(df, protected_attrs, label_name, use_protected, use_sample_weight=False, log='log.txt', model='gbm', num_iter=10, rand_seed=42):
    base_acc = np.zeros((num_iter,))
    base_meo = np.zeros((num_iter,))
    base_deo = np.zeros((num_iter,))
    base_sp = np.zeros((num_iter,))

    leverage_acc = np.zeros((num_iter,))
    leverage_meo = np.zeros((num_iter,))
    leverage_deo = np.zeros((num_iter,))
    leverage_sp = np.zeros((num_iter,))
    leverage_time = np.zeros((num_iter,))

    t_all = time.localtime()
    log = open(log, 'w+')
    
    for seed in tqdm(range(num_iter)):
        log.write('Iteration: {:2d}/{:2d}\n'.format(seed + 1, num_iter))
        log.flush()
        t_epoch = time.localtime()
        ## train/test split using aif360.datasets.StandardDatasets
        dataset_orig_train, dataset_orig_test = train_test_split(df, test_size=0.3, random_state=seed)

        dataset_orig_train = StandardDataset(dataset_orig_train, label_name=label_name, favorable_classes=[1],
                                             protected_attribute_names=protected_attrs, privileged_classes=[[1]])
        dataset_orig_test = StandardDataset(dataset_orig_test, label_name=label_name, favorable_classes=[1],
                                            protected_attribute_names=protected_attrs, privileged_classes=[[1]])

        if use_protected:
            idx_features = get_idx_w_protected(dataset_orig_train.feature_names)
        else:
            idx_features = get_idx_wo_protected(dataset_orig_train.feature_names, protected_attrs)
        idx_protected = get_idx_protected(dataset_orig_train.feature_names, protected_attrs)

        X_train, y_train = dataset_orig_train.features[:, idx_features], dataset_orig_train.labels.ravel()
        X_test, y_test = dataset_orig_test.features[:, idx_features], dataset_orig_test.labels.ravel()
        s_train = dataset_orig_train.features[:, idx_protected].ravel()
        s_test = dataset_orig_test.features[:, idx_protected].ravel()

        # declare classifiers
        if model == 'gbm':
            clf_YgX = GradientBoostingClassifier(random_state=rand_seed)  # will predict Y from X
        elif model == 'logit':
            clf_YgX = LogisticRegression(random_state=rand_seed)  # will predict Y from X
        elif model == 'rf':
            clf_YgX = RandomForestClassifier(random_state=rand_seed, n_estimators=500, min_samples_leaf=10)  # will predict Y from X
        else:
            log.write('Error: Undefined Model\n')
            log.flush()
            return

        ## train base classifier
        clf_YgX.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)
        base_acc[seed], base_meo[seed], base_deo[seed], base_sp[seed] = evaluate_leverage(X_test, y_test, clf_YgX.predict(X=X_test), s_test)

        ## perform leveraging approach
        lev_time = time.localtime()
        ## compute prior
        y_test_prob = clf_YgX.predict_proba(X=X_test)[:, 1]
        female_msk = (s_test == 0)
        E_X0 = y_test_prob[female_msk].mean()
        E_X1 = y_test_prob[~female_msk].mean()
        P_Y0S0 = y_test_prob[female_msk].mean() * female_msk.mean()
        P_Y0S1 = y_test_prob[female_msk].mean() * (~female_msk).mean()

        ## compute theta
        objective = np.inf
        for theta in np.concatenate((-np.logspace(-2, 2, 10000), np.logspace(-2, 2, 10000))):
            m0 = (y_test_prob[female_msk] * (y_test_prob[female_msk] * (2.0 - theta / P_Y0S0) >= 1)).mean() / E_X0
            m1 = (y_test_prob[~female_msk] * (y_test_prob[~female_msk] * (2.0 + theta / P_Y0S1) >= 1)).mean() / E_X1

            tmp = np.abs(m0 - m1)
            if objective > tmp:
                objective = tmp
                thetahat = theta

        ## apply theta
        ntest = len(y_test)
        y_test_leveraged = np.zeros((ntest,))
        for i in range(ntest):
            if s_test[i] == 0:
                y_test_leveraged[i] = y_test_prob[i] * (2.0 - thetahat / P_Y0S0) >= 1
            else:
                y_test_leveraged[i] = y_test_prob[i] * (2.0 + thetahat / P_Y0S1) >= 1

        leverage_time[seed] = (time.mktime(time.localtime()) - time.mktime(lev_time)) / 60

        leverage_acc[seed], leverage_meo[seed], leverage_deo[seed], leverage_sp[seed] = evaluate_leverage(X_test, y_test, y_test_leveraged, s_test)

        log.write(' Base: Acc {:.2f}, meo {:.2f}, deo {:.2f}, sp {:.2f}\n'.format(base_acc[seed], base_meo[seed], base_deo[seed], base_sp[seed]))
        log.write(' Leve: Acc {:.2f}, meo {:.2f}, deo {:.2f}, sp {:.2f}\n'.format(leverage_acc[seed], leverage_meo[seed], leverage_deo[seed], leverage_sp[seed]))
        log.write(' Time : {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_epoch)) / 60))
        log.flush()

    base = {
        'acc': base_acc,
        'meo': base_meo,
        'deo': base_deo,
        'sp': base_sp
    }
    leverage = {
        'acc': leverage_acc,
        'meo': leverage_meo,
        'deo': leverage_deo,
        'sp': leverage_sp,
        'time': leverage_time
    }
    log.write(' Total Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_all))))
    log.flush()
    return base, leverage, leverage_time



def MP_tol(df, tolerance, protected_attrs, label_name, use_protected, use_sample_weight, tune_threshold, log, model='gbm', div='cross-entropy', num_iter=10, rand_seed=42, constraint='meo'):
    acc = np.zeros((len(tolerance), num_iter))
    brier = np.zeros((len(tolerance), num_iter))
    auc = np.zeros((len(tolerance), num_iter))
    meo = np.zeros((len(tolerance), num_iter))
    meo_abs = np.zeros((len(tolerance), num_iter))
    mo = np.zeros((len(tolerance), num_iter))
    sp = np.zeros((len(tolerance), num_iter))
    dcp_msk = np.zeros((len(tolerance), num_iter))
    # protected_attrs = ['racebin']
    # label_name = 'gradebin'

    t_all = time.localtime()
    for seed in tqdm(range(num_iter)):
        log.write(' Iteration: {:2d}/{:2d}\n'.format(seed+1, num_iter))
        log.flush()
        t_epoch = time.localtime()
        ## train/test split using aif360.datasets.StandardDatasets
        dataset_orig_train, dataset_orig_test = train_test_split(df, test_size=0.3, random_state=seed)

        dataset_orig_train = StandardDataset(dataset_orig_train, label_name=label_name, favorable_classes=[1],
                                             protected_attribute_names=protected_attrs, privileged_classes=[[1]])
        dataset_orig_test = StandardDataset(dataset_orig_test, label_name=label_name, favorable_classes=[1],
                                            protected_attribute_names=protected_attrs, privileged_classes=[[1]])

        if use_protected:
            idx_features = get_idx_w_protected(dataset_orig_train.feature_names)
        else:
            idx_features = get_idx_wo_protected(dataset_orig_train.feature_names, protected_attrs)
        idx_protected = get_idx_protected(dataset_orig_train.feature_names, protected_attrs)

        X_train, y_train = dataset_orig_train.features[:, idx_features], dataset_orig_train.labels.ravel()
        X_test, y_test = dataset_orig_test.features[:, idx_features], dataset_orig_test.labels.ravel()
        s_train = dataset_orig_train.features[:, idx_protected].ravel()
        s_test = dataset_orig_test.features[:, idx_protected].ravel()

        # declare classifiers
        if model == 'gbm':
            clf_YgX = GradientBoostingClassifier(random_state=rand_seed)  # will predict Y from X
            clf_SgX = GradientBoostingClassifier(random_state=rand_seed)  # will predict S from X (needed for SP)
            clf_SgXY = GradientBoostingClassifier(random_state=rand_seed)  # will predict S from (X,Y)
        elif model == 'logit':
            clf_YgX = LogisticRegression(random_state=rand_seed)  # will predict Y from X
            clf_SgX = LogisticRegression(random_state=rand_seed)  # will predict S from X (needed for SP)
            clf_SgXY = LogisticRegression(random_state=rand_seed)  # will predict S from (X,Y)
        elif model == 'rfc':
            clf_YgX = RandomForestClassifier(random_state=rand_seed, n_estimators=10, min_samples_leaf=10)  # will predict Y from X
            clf_SgX = RandomForestClassifier(random_state=rand_seed, n_estimators=10, min_samples_leaf=10)  # will predict S from X (needed for SP)
            clf_SgXY = RandomForestClassifier(random_state=rand_seed, n_estimators=10, min_samples_leaf=10)  # will predict S from (X,Y)
        else:
            log.write('Error: Undefined Model\n')
            log.flush()
            return

        t_fit = time.localtime()
        ## initalize GFair class and train classifiers
        gf = GF.GFair(clf_YgX, clf_SgX, clf_SgXY, div=div)
        if use_sample_weight:
            gf.fit(X=X_train, y=y_train, s=s_train, sample_weight=dataset_orig_train.instance_weights)
        else:
            gf.fit(X=X_train, y=y_train, s=s_train, sample_weight=None)
        log.write('  Time to fit the base models: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_fit))/60))
        log.flush()

        ## start projection
        for i, tol in enumerate(tolerance):
            t_tol = time.localtime()

            try: ## in case the solver has issues
                ## model projection
                constraints = [(constraint, tol)]
                gf.project(X=X_train, s=s_train, constraints=constraints, rho=2, max_iter=500, method='tf')

                log.write('  Tolerance: {:.4f}, projection time: {:4.3f} mins, '.format(tol, (time.mktime(time.localtime()) - time.mktime(t_tol)) / 60))
                log.flush()

                ## set classification threshold
                t_threshold = time.localtime()
                if not tune_threshold:
                    threshold = 0.5
                else:
                    threshold = search_threshold(gf, X_train, y_train, s_train)
                log.write('threshold: {:.4f}, threshold time: {:4.3f} mins\n'.format(threshold, (time.mktime(time.localtime()) - time.mktime(t_threshold)) / 60))
                log.flush()

                ## evaluation
                evaluation(i, seed, gf, X_test, y_test, s_test, threshold,
                           acc, brier, auc, meo, meo_abs, mo, sp)

            except:
                dcp_msk[i, seed] = 1
                log.write('  Tolerance: {:.4f}, DCPError!!!\n'.format(tol))
                log.flush()
                continue

        log.write('  Epoch Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_epoch))/60))
        log.flush()

    results = {
        'acc': acc,
        'brier': brier,
        'auc': auc,
        'meo': meo,
        'meo_abs': meo_abs,
        'mo': mo,
        'sp': sp,
        'dcp': dcp_msk
    }
    log.write(' Total Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_all))/60))
    log.flush()
    return results
