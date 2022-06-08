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
    elif name == 'compas':
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
        
        
    elif name == 'hsls': 
        file_path = '../data/HSLS/'
        filename = 'hsls_knn_impute.pkl'
        df = pd.read_pickle(file_path+filename)

        # ## if no variables specified, include all variables
        # if vars != []:
        #     df = df[vars]

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
        
    elif name == 'enem':
        file_path = '../data/ENEM/'
        filename = 'enem-50000-20.pkl'
        df = pd.read_pickle(file_path+filename)
        df['gradebin'] = df['gradebin'].astype(int)
        
    # TODO: add other datasets here
        
    return df

#%% method for computing fitting models on dataset. Returns P_{Y,S|X}, P_{Y|X}, P_{S|X}, and P_{S,Y}
class clf:
    
    def __init__(self,df,model_dict,S=[],Y=[],X=[]):
        """Initialize the predictive model class.
        
        This function receives a dataframe df, a dictionary of scikitlearn model (must have fit/predict functionality).
        It also takes a list of variablenames for S, Y, and X.
        The variable names must be valid column names of df.
        """
        
        # train values -- converting to cateagorical
        
        # create categorical dictionary and labels for YS
        self.YSdict = self.create_dict(df,Y+S) 
        ys_train = [self.YSdict[tuple(x)] for x in df[Y+S].values]
        
        
        # create categorical dictionary for Y
        if len(Y) == 1:
             y_train = df[Y].values
        else: 
            self.Ydict = self.create_dict(df,Y)
            y_train = [self.Ydict[tuple(x)] for x in df[Y].values]
        
        # create categorical dictionary for S
        if len(S) == 1:
            s_train = df[S].values
        else:
            self.Sdict = self.create_dict(df,S) 
            s_train = [self.Sdict[tuple(x)] for x in df[S].values]
        
        # declare models --- add new models and change parameters here!
        self.mPys_x = model_dict['Pys_x']
        self.mPy_x  = model_dict['Py_x']
        self.mPs_x = model_dict['Ps_x']
        
        #%%% fit models
        # Pys_x
        self.mPys_x.fit(df[X],ys_train)
     
        ys_predict = self.mPys_x.predict_proba(df[X])
        print('Training acc Pys_x: ' + str(sm.accuracy_score(np.argmax(ys_predict,axis=1),ys_train)))
        
        # Py_x
        self.mPy_x.fit(df[X],y_train)
        
        y_predict = self.mPy_x.predict_proba(df[X])
        print('Training acc Py_x: ' + str(sm.accuracy_score(np.argmax(y_predict,axis=1),y_train)))
        
        # Ps_x
        self.mPs_x.fit(df[X],s_train)
        
        s_predict = self.mPs_x.predict_proba(df[X])
        print('Training acc Ps_x: ' + str(sm.accuracy_score(np.argmax(s_predict,axis=1),s_train)))
        
        
        
        # print training errors
        
        
        # compute marginals
        self.mPys = df.groupby(Y+S).size().unstack(S)/len(df)
        
    #helper function for creating categorical dict
    def create_dict(self,df,features):
        featureDict = df.groupby(features).size()
        featureDict[:] = range(len(featureDict))
        featureDict = featureDict.to_dict()
        return featureDict
        

        
        
        
    
