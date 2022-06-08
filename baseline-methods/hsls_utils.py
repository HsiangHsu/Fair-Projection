import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import confusion_matrix





def load_hsls(file_path, filename, vars):
    ## group_feature can be either 'sexbin' or 'racebin'
    ## load pickle
    df = pd.read_csv(file_path+filename)

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
    df['gradebin'] = (df['X1TXMSCR'] > df['X1TXMSCR'].median()).astype(int)
    df['racebin'] = np.logical_or((df['X1RACE'] == 8).values, (df['X1RACE'] == 2).values).astype(int)
    df['sexbin'] = (df['X1SEX'] == 1).astype(int)


    ## Dropping race and 12th grade data just to focus on the 9th grade prediction ##
    df = df.drop(columns=['X1SEX', 'X1RACE', 'X1TXMSCR', 'X2TXMSCR'])

    ## Scaling ##
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    ## Balancing data to have roughly equal race=0 and race =1 ##
    # df = balance_data(df, group_feature)
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


