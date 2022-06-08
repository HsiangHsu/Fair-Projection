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


def construct_grade(df, grade_attribute, n):
    v = df[grade_attribute[0]].values
    quantiles = np.nanquantile(v, np.linspace(0.0, 1.0, n+1))
    return pd.cut(v, quantiles, labels=np.arange(n))

def construct_race(df, protected_attribute):
    race_dict = {'Branca': 1, 'Preta': 2, 'Parda': 3, 'Amarela': 4, 'Indigena': 5} # changed to match ENEM 2020 numbering
    return df[protected_attribute].map(race_dict)

def load_enem20(file_path, filename, features, grade_attribute, n_sample, n_classes, multigroup=False):
    ## load csv
    df = pd.read_csv(file_path+filename, encoding='cp860', sep=';')
    # print('Original Dataset Shape:', df.shape)

    ## Remove all entries that were absent or were eliminated in at least one exam
    ix = ~df[['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT']].applymap(lambda x: False if x == 1.0 else True).any(axis=1)
    df = df.loc[ix, :]

    ## Remove "treineiros" -- these are individuals that marked that they are taking the exam "only to test their knowledge". It is not uncommon for students to take the ENEM in the middle of high school as a dry run
    df = df.loc[df['IN_TREINEIRO'] == 0, :]

    ## drop eliminated features
    df.drop(['TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT', 'IN_TREINEIRO'], axis=1, inplace=True)

    ## subsitute race by names
    # race_names = ['N/A', 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    race_names = [np.nan, 'Branca', 'Preta', 'Parda', 'Amarela', 'Indigena']
    df['TP_COR_RACA'] = df.loc[:, ['TP_COR_RACA']].applymap(lambda x: race_names[x]).copy()

    ## remove repeated exam takers
    ## This pre-processing step significantly reduces the dataset.
    # df = df.loc[df.TP_ST_CONCLUSAO.isin([1,2])]
    df = df.loc[df.TP_ST_CONCLUSAO.isin([1])] 

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variable
    df['gradebin'] = construct_grade(df, grade_attribute, n_classes)
    if multigroup:
        df['racebin'] = construct_race(df, 'TP_COR_RACA')
    else:
        df['racebin'] =np.logical_or((df['TP_COR_RACA'] == 'Branca').values, (df['TP_COR_RACA'] == 'Amarela').values).astype(int)
    df['sexbin'] = (df['TP_SEXO'] == 'M').astype(int)

    df.drop([grade_attribute[0], 'TP_COR_RACA', 'TP_SEXO'], axis=1, inplace=True)

    ## encode answers to questionaires
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ['Q00' + str(x) if x < 10 else 'Q0' + str(x) for x in range(1, 25)]
    for q in question_vars:
        if q != 'Q005':
            df_q = pd.get_dummies(df[q], prefix=q)
            df.drop([q], axis=1, inplace=True)
            df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)
            
    ## check if age range ('TP_FAIXA_ETARIA') is within attributes
    if 'TP_FAIXA_ETARIA' in features:
        q = 'TP_FAIXA_ETARIA'
        df_q = pd.get_dummies(df[q], prefix=q)
        df.drop([q], axis=1, inplace=True)
        df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## encode SG_UF_PROVA (state where exam was taken)
    df_res = pd.get_dummies(df['SG_UF_PROVA'], prefix='SG_UF_PROVA')
    df.drop(['SG_UF_PROVA'], axis=1, inplace=True)
    df = pd.concat([df, df_res], axis=1)

    df = df.dropna()
    ## Scaling ##
    scaler = MinMaxScaler()
    scale_columns = list(set(df.columns.values) - set(['gradebin', 'racebin']))
    df[scale_columns] = pd.DataFrame(scaler.fit_transform(df[scale_columns]), columns=scale_columns, index=df.index)
    # print('Preprocessed Dataset Shape:', df.shape)

    df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)
    df['gradebin'] = df['gradebin'].astype(int)
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

    return (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, max(np.abs(tpr_diff), np.abs(fpr_diff))

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

def MP_tol(df, tolerance, use_protected, use_sample_weight, tune_threshold, log, model='gbm', div='cross-entropy', num_iter=10, rand_seed=42, constraint='meo', projection_method='np'):
    acc = np.zeros((len(tolerance), num_iter))
    brier = np.zeros((len(tolerance), num_iter))
    auc = np.zeros((len(tolerance), num_iter))
    meo = np.zeros((len(tolerance), num_iter))
    meo_abs = np.zeros((len(tolerance), num_iter))
    mo = np.zeros((len(tolerance), num_iter))
    sp = np.zeros((len(tolerance), num_iter))
    # dcp_msk = np.zeros((len(tolerance), num_iter))
    fittime = np.zeros((num_iter))
    projectionime = np.zeros((len(tolerance), num_iter))
    evaluationtime = np.zeros((len(tolerance), num_iter))
    epochtime = np.zeros((num_iter))

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
        fit_length = (time.mktime(time.localtime()) - time.mktime(t_fit))/60
        log.write('  Time to fit the base models: {:4.3f} mins\n'.format(fit_length))
        log.flush()
        fittime[seed] = fit_length
        ## start projection
        for i, tol in enumerate(tolerance):


            # try: ## in case the solver has issues
            ## model projection
            t_projection = time.localtime()
            constraints = [(constraint, tol)]
            gf.project(X=X_train, s=s_train, constraints=constraints, rho=2, max_iter=500, method=projection_method)

            projection_length = (time.mktime(time.localtime()) - time.mktime(t_projection)) / 60
            log.write('  Tolerance: {:.4f}, projection time: {:4.3f} mins, '.format(tol, projection_length))
            log.flush()
            projectionime[i, seed] = projection_length
            ## set classification threshold
            t_threshold = time.localtime()
            if not tune_threshold:
                threshold = 0.5
            else:
                threshold = search_threshold(gf, X_train, y_train, s_train)
            log.write('threshold: {:.4f}, threshold time: {:4.3f} mins\n'.format(threshold, (time.mktime(time.localtime()) - time.mktime(t_threshold)) / 60))
            log.flush()

            ## evaluation
            t_evaluation = time.localtime()
            evaluation(i, seed, gf, X_test, y_test, s_test, threshold,
                       acc, brier, auc, meo, meo_abs, mo, sp)
            evaluation_length = (time.mktime(time.localtime()) - time.mktime(t_evaluation)) / 60
            evaluationtime[i, seed] = evaluation_length
            # except:
            #     dcp_msk[i, seed] = 1
            #     log.write('  Tolerance: {:.4f}, DCPError!!!\n'.format(tol))
            #     log.flush()
            #     continue
        epoch_length = (time.mktime(time.localtime()) - time.mktime(t_epoch))/60
        log.write('  Epoch Time: {:4.3f} mins\n'.format(epoch_length))
        log.flush()
        epochtime[seed] = epoch_length

    totaltime = (time.mktime(time.localtime()) - time.mktime(t_all))/60
    log.write(' Total Time: {:4.3f} mins\n'.format(totaltime))
    log.flush()
    results = {
        'acc': acc,
        'brier': brier,
        'auc': auc,
        'meo': meo,
        'meo_abs': meo_abs,
        'mo': mo,
        'sp': sp,
        # 'dcp': dcp_msk,
        'fittime': fittime,
        'projectionime': projectionime,
        'evaluationtime': evaluationtime,
        'epochtime': epochtime,
        'totaltime': totaltime
    }


    return results
