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

## custom packages
import DataLoader as DL
import coreMP as MP
import GroupFair as GF

def load_hsls_imputed(file_path, filename, vars):
    ## group_feature can be either 'sexbin' or 'racebin'
    ## load csv
    df = pd.read_pickle(file_path+filename)

    ## if no variables specified, include all variables
    if vars != []:
        df = df[vars]

    ## Setting NaNs to out-of-range entries
    ## entries with values smaller than -7 are set as NaNs
    df[df <= -7] = np.nan

    ## Dropping all rows or columns with missing values
    ## this step significantly reduces the number of samples
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variables
    ## X1SEX: 1 -- Male, 2 -- Female, -9 -- NaN -> Preprocess it to: 0 -- Female, 1 -- Male, drop NaN
    ## X1RACE: 0 -- BHN, 1 -- WA
    df['gradebin'] = df['grade9thbin']
    df['racebin'] = np.logical_or(((df['studentrace']*7).astype(int)==7).values, ((df['studentrace']*7).astype(int)==1).values).astype(int)
    df['sexbin'] = df['studentgender'].astype(int)


    ## Dropping race and 12th grade data just to focus on the 9th grade prediction ##
    df = df.drop(columns=['studentgender', 'grade9thbin', 'grade12thbin', 'studentrace'])

    ## Scaling ##
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    ## Balancing data to have roughly equal race=0 and race =1 ##
    # df = balance_data(df, group_feature)
    return df

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

    return (tpr_diff + fpr_diff) / 2, (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, max(np.abs(tpr_diff), np.abs(fpr_diff))

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

def evaluation(idx1, idx2, clf, X, y, s, t,
               acc, brier, auc, meo, meo_abs, mo, sp):
    y_prob = np.squeeze(clf.predict_proba(X=X, s=s), axis=2)
    y_pred = (y_prob[:, 1] > t).astype('int')

    acc[idx1, idx2] = accuracy_score(y, y_pred)
    brier[idx1, idx2] = brier_score_loss(y, y_prob[:, 1])
    auc[idx1, idx2] = roc_auc_score(y_true=y, y_score=y_prob[:, 1])
    meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs(y, y_pred, s)
    sp[idx1, idx2] = statistical_parity_difference(y_pred, s)
    return

def MP_tol(df, tolerance, use_protected, use_sample_weight, tune_threshold, log, model='gbm', div='cross-entropy', num_iter=10, rand_seed=42, constraint='meo'):
    acc = np.zeros((len(tolerance), num_iter))
    brier = np.zeros((len(tolerance), num_iter))
    auc = np.zeros((len(tolerance), num_iter))
    meo = np.zeros((len(tolerance), num_iter))
    meo_abs = np.zeros((len(tolerance), num_iter))
    mo = np.zeros((len(tolerance), num_iter))
    sp = np.zeros((len(tolerance), num_iter))
    dcp_msk = np.zeros((len(tolerance), num_iter))
    protected_attrs = ['racebin']
    label_name = 'gradebin'

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
