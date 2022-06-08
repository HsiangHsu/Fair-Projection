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
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, multilabel_confusion_matrix

from scipy.special import kl_div
from itertools import combinations

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

def load_enem(file_path, filename, features, grade_attribute, n_sample, n_classes, multigroup=False):
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
    # df['gradebin'] = df['gradebin'].astype(int)
    return df

def confusion(y, y_pred):
    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

def multilabel_confusion(y, y_pred, nc):
    cm = multilabel_confusion_matrix(y, y_pred)  ## (nc, 2, 2) ##
    print(cm.shape)
    print(y.shape, y_pred.shape)
    #     cm = confusion_matrix(y, y_pred)
    #     print(cm)
    tprs, fprs = np.zeros((nc)), np.zeros((nc))
    for i in range(nc):
        tn, fp, fn, tp = cm[i, :, :].ravel()

        tprs[i] = tp / (tp + fn)
        fprs[i] = fp / (fp + tn)
    return tprs, fprs

def odd_diffs_binary(y, y_pred, s):
    y0, y1 = y[s==1], y[s==2]
    y_pred0, y_pred1 = y_pred[s==1], y_pred[s==2]

    tpr0, fpr0 = confusion(y0, y_pred0)
    tpr1, fpr1 = confusion(y1, y_pred1)

    tpr_diff = tpr1 - tpr0
    fpr_diff = fpr1 - fpr0

    return (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, (np.abs(tpr_diff) + np.abs(fpr_diff)) / 2, max(np.abs(tpr_diff), np.abs(fpr_diff))

def odd_diffs_multi(y, y_pred, s, ns, nc):
    ## for binary case,
    tpr_diff, fpr_diff = np.zeros((nc, ns * (ns - 1) // 2)), np.zeros((nc, ns * (ns - 1) // 2))
    tprs, fprs = np.zeros((ns, nc)), np.zeros((ns, nc))
    print('s = ', s.shape, list(set(s)))
    for i in range(ns):
        y_s = y[s == (i+1)]
        y_pred_s = y_pred[s == (i+1)]

        tprs[i, :], fprs[i, :] = multilabel_confusion(y_s, y_pred_s, nc)

    for i in range(nc):
        tpr_diff[i, :] = np.array([a1 - a2 for (a1, a2) in combinations(tprs[:, i], 2)])
        fpr_diff[i, :] = np.array([a1 - a2 for (a1, a2) in combinations(fprs[:, i], 2)])


    meo = (np.abs(tpr_diff) + np.abs(fpr_diff) / (2 * ns) * nc).max()
    meo_abs = np.abs((tpr_diff + fpr_diff) / ns).max()
    mo = np.max(np.maximum(np.abs(tpr_diff), np.abs(fpr_diff)))
    return meo, meo_abs, mo

def statistical_parity_binary(y, s):
    sp0 = y[s==0].mean()
    sp1 = y[s==1].mean()
    return np.abs(sp1-sp0)

def statistical_parity_multi(y, s, ns, nc):
    ## sp_{i, j} = y[s==i, y==j].sum() / len(y[s==i]) = Pr (Y= j |S =i), 1<=i<=ns, 1<=j<=nc
    sp = np.zeros((ns, nc))
    for i in range(ns):
        for j in range(nc):
            sp[i, j] = len(y[np.logical_and(s == (i+1), y == j)]) / len(y[s == (i+1)])

    sp_class = []
    for j in range(nc):
        sp_class.append(max([np.abs(a1 - a2) for (a1, a2) in combinations(sp[:, j], 2)]))

    return max(sp_class) * (nc/ns)

def evaluation(idx1, idx2, clf, X, y, s, y_base, ns, nc,
               acc, kl, logloss, meo, meo_abs, mo, sp):
    y_prob = np.squeeze(clf.predict_proba(X=X, s=s), axis=2)
    y_pred = y_prob.argmax(axis=1)

    acc[idx1, idx2] = accuracy_score(y, y_pred)
    kl[idx1, idx2] = kl_div(y_prob, y_base).mean()
    logloss[idx1, idx2] = kl_div(y_base, y_prob).mean()

    # meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs_multi(y, y_pred, s, ns, nc)
    # sp[idx1, idx2] = statistical_parity_multi(y_pred, s, ns, nc)
    if nc == 2 and ns == 2:
        meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs_binary(y, y_pred, s)
        sp[idx1, idx2] = statistical_parity_binary(y_pred, s)
    elif nc == 2 and ns > 2:
        meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs_multi(y, y_pred, s, ns, nc)
        sp[idx1, idx2] = statistical_parity_multi(y_pred, s, ns, nc)
    elif nc > 2 and ns == 2:
        meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs_multi(y, y_pred, s, ns, nc)
        sp[idx1, idx2] = statistical_parity_multi(y_pred, s, ns, nc)
    else:
        meo[idx1, idx2], meo_abs[idx1, idx2], mo[idx1, idx2] = odd_diffs_multi(y, y_pred, s, ns, nc)
        sp[idx1, idx2] = statistical_parity_multi(y_pred, s, ns, nc)
    return

def MP_tol(df, ns, nc, tolerance, use_protected, log, model='gbm', div='cross-entropy', num_iter=10, rand_seed=42, constraint='meo'):
    acc = np.zeros((len(tolerance), num_iter))
    kl = np.zeros((len(tolerance), num_iter))
    logloss = np.zeros((len(tolerance), num_iter))
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
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=seed)
        if use_protected:
            feature_names = list(set(df.columns.values) - set([label_name]))
        else:
            feature_names = list(set(df.columns.values) - set([label_name, protected_attrs[0]]))
        X_train, y_train = df_train.loc[:, feature_names].values, np.asarray(df_train[label_name].values.ravel())
        X_test, y_test = df_test.loc[:, feature_names].values, np.asarray(df_test[label_name].values.ravel())
        s_train, s_test = df_train[protected_attrs[0]].values.ravel(), df_test[protected_attrs[0]].values.ravel()

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
        gf.fit(X=X_train, y=y_train, s=s_train, sample_weight=None)
        log.write('  Time to fit the base models: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_fit))/60))
        log.flush()

        y_prob_base = clf_YgX.predict_proba(X_test)

        ## start projection
        for i, tol in enumerate(tolerance):
            t_tol = time.localtime()

            # try: ## in case the solver has issues
            ## model projection
            constraints = [(constraint, tol)]
            gf.project(X=X_train, s=s_train, constraints=constraints, rho=2, max_iter=500, method='tf')

            log.write('  Tolerance: {:.4f}, projection time: {:4.3f} mins\n'.format(tol, (time.mktime(time.localtime()) - time.mktime(t_tol)) / 60))
            log.flush()

            ## evaluation
            print(ns, nc, nc==5)
            print(list(set(y_test)))
            evaluation(i, seed, gf, X_test, y_test, s_test, y_prob_base, ns, nc,
                       acc, kl, logloss, meo, meo_abs, mo, sp)
            

            # except:
            #     dcp_msk[i, seed] = 1
            #     log.write('  Tolerance: {:.4f}, Does not convergence!!!\n'.format(tol))
            #     log.flush()
            #     continue

        log.write('  Epoch Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_epoch))/60))
        log.flush()

    results = {
        'acc': acc,
        'kl': kl,
        'logloss': logloss,
        'meo': meo,
        'meo_abs': meo_abs,
        'mo': mo,
        'sp': sp,
        'dcp': dcp_msk
    }
    log.write(' Total Time: {:4.3f} mins\n'.format((time.mktime(time.localtime()) - time.mktime(t_all))/60))
    log.flush()
    return results
