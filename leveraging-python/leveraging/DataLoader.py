#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:01:09 2020

@author: flavio
"""

import pandas as pd
import numpy as np
from sklearn import metrics as sm
import sys

from sklearn.preprocessing import MinMaxScaler
#%% method for loading different datasets
def load_data(name='adult'):
    
    #% Processing for UCI-ADULT
    if name == 'adult':
        file = '../data/UCI-Adult/adult.data'
        fileTest = '../data/UCI-Adult/adult.test'
        
        df = pd.read_csv(file, header=None,sep=',\s+',engine='python')
        dfTest = pd.read_csv(fileTest,header=None,skiprows=1,sep=',\s+',engine='python') 
        
        
        columnNames = ["age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        df.columns = columnNames
        dfTest.columns = columnNames
        
        df = df.append(dfTest)
        
        # drop columns that won't be used
        dropCol = ["fnlwgt","workclass","occupation"]
        df.drop(dropCol,inplace=True,axis=1)
        
        # keep only entries marked as ``White'' or ``Black''
        ix = df['race'].isin(['White','Black'])
        df = df.loc[ix,:]
        
        # binarize race
        # Black = 0; White = 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='White' else 0)
        
        # binarize gender
        # Female = 0; Male = 1
        df.loc[:,'gender'] = df['gender'].apply(lambda x: 1 if x=='Male' else 0)
        
        # binarize income
        # '>50k' = 1; '<=50k' = 0
        df.loc[:,'income'] = df['income'].apply(lambda x: 1 if x[0]=='>' else 0)
        
        
        # drop "education" and native-country (education already encoded in education-num)
        features_to_drop = ["education","native-country"]
        df.drop(features_to_drop,inplace=True,axis=1)
        
        
        
        # create one-hot encoding
        categorical_features = list(set(df)-set(df._get_numeric_data().columns))
        df = pd.concat([df,pd.get_dummies(df[categorical_features])],axis=1,sort=False)
        df.drop(categorical_features,inplace=True,axis=1)
        
        # reset index
        df.reset_index(inplace=True,drop=True)
        
    
    #% Processing for COMPAS
    if name == 'compas':
        file = '../data/COMPAS/compas-scores-two-years.csv'
        df = pd.read_csv(file,index_col=0)
        
        # select features for analysis
        df = df[['age', 'c_charge_degree', 'race',  'sex', 'priors_count', 
                    'days_b_screening_arrest',  'is_recid',  'c_jail_in', 'c_jail_out']]
        
        # drop missing/bad features (following ProPublica's analysis)
        # ix is the index of variables we want to keep.

        # Remove entries with inconsistent arrest information.
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix

        # remove entries entries where compas case could not be found.
        ix = (df['is_recid'] != -1) & ix

        # remove traffic offenses.
        ix = (df['c_charge_degree'] != "O") & ix


        # trim dataset
        df = df.loc[ix,:]

        # create new attribute "length of stay" with total jail time.
        df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)
        
        
        # drop 'c_jail_in' and 'c_jail_out'
        # drop columns that won't be used
        dropCol = ['c_jail_in', 'c_jail_out','days_b_screening_arrest']
        df.drop(dropCol,inplace=True,axis=1)
        
        # keep only African-American and Caucasian
        df = df.loc[df['race'].isin(['African-American','Caucasian']),:]
        
        # binarize race 
        # African-American: 0, Caucasian: 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='Caucasian' else 0)
        
        # binarize gender
        # Female: 1, Male: 0
        df.loc[:,'sex'] = df['sex'].apply(lambda x: 1 if x=='Male' else 0)
        
        # rename columns 'sex' to 'gender'
        df.rename(index=str, columns={"sex": "gender"},inplace=True)
        
        # binarize degree charged
        # Misd. = -1, Felony = 1
        df.loc[:,'c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 1 if x=='F' else -1)
               
        # reset index
        df.reset_index(inplace=True,drop=True)
        
        
    # TODO: add other datasets here
        
    return df

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

def load_enem(file_path, filename, features, grade_attribute, n_sample):
    ## load csv
    df = pd.read_csv(file_path+filename, encoding='cp860', sep=';')
    print('Original Dataset Shape:', df.shape)

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
    df = df.loc[df.TP_ST_CONCLUSAO == 2]

    ## select features
    df = df[features]

    ## Dropping all rows or columns with missing values
    df = df.dropna()

    ## Creating racebin & gradebin & sexbin variable
    df['gradebin'] = (df[grade_attribute[0]] > df[grade_attribute[0]].median()).astype(int)
    df['racebin'] = np.logical_or((df['TP_COR_RACA'] == 'Branca').values,
                                  (df['TP_COR_RACA'] == 'Amarela').values).astype(int)
    df['sexbin'] = (df['TP_SEXO'] == 'M').astype(int)

    df.drop([grade_attribute[0], 'TP_COR_RACA', 'TP_SEXO'], axis=1, inplace=True)

    ## encode answers to questionaires
    ## Q005 is 'Including yourself, how many people currently live in your household?'
    question_vars = ['Q00' + str(x) if x < 10 else 'Q0' + str(x) for x in range(1, 28)]
    for q in question_vars:
        if q != 'Q005':
            df_q = pd.get_dummies(df[q], prefix=q)
            df.drop([q], axis=1, inplace=True)
            df = pd.concat([df, df_q.iloc[:, :-1]], axis=1)

    ## encode SG_UF_RESIDENCIA
    df_res = pd.get_dummies(df['SG_UF_RESIDENCIA'], prefix='SG_UF_RESIDENCIA')
    df.drop(['SG_UF_RESIDENCIA'], axis=1, inplace=True)
    df = pd.concat([df, df_res], axis=1)

    ## Scaling ##
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    print('Preprocessed Dataset Shape:', df.shape)

    df = df.sample(n=min(n_sample, df.shape[0]), axis=0, replace=False)
    return df
        

        
        
        
    
