import numpy as np
from sklearn.model_selection import train_test_split
import os, urllib
from .helper import *
from .plot import *
from .fairness import *
from sklearn.preprocessing import MinMaxScaler

def get_dataset(name, save=False, corr_sens=False, seed=42, verbose=False):
    """
    Retrieve dataset and all relevant information
    :param name: name of the dataset
    :param save: if set to True, save the dataset as a pickle file. Defaults to False
    :return: Preprocessed dataset and relevant information
    """
    def get_numpy(df):
        new_df = df.copy()
        cat_columns = new_df.select_dtypes(['category']).columns
        new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
        return new_df.values

    if name == 'adult':
        # Load data
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', \
                         'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
        df = pd.read_csv('../dataset/adult.data', names=feature_names)
        if verbose:
            print('Raw Dataset loaded.')
        num_train = df.shape[0]
        pos_class_label = ' >50K'
        neg_class_label = ' <=50K'
        y = np.zeros(num_train)
        y[df.iloc[:,-1].values == pos_class_label] = 1
        df = df.drop(['fnlwgt', 'education-num'], axis=1)
        num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        feature_names = num_var_names + cat_var_names
        df = df[feature_names]
        if verbose:
            print('Selecting relevant features complete.')

        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        dtypes = df.dtypes

        X = get_numpy(df)
        if verbose:
            print('Numpy conversion complete.')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # sens idx
        race_idx = 9
        sex_idx = 10
        sens_idc = [sex_idx]

        race_cats = df[feature_names[race_idx]].cat.categories
        sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')

    elif name == 'compas':
        file = '../../data/COMPAS/compas-scores-two-years.csv'
        df = pd.read_csv(file,index_col=0)
        
        df = df[['age', 'c_charge_degree', 'race',  'sex', 'priors_count', 
                    'days_b_screening_arrest',  'is_recid',  'c_jail_in', 'c_jail_out']]
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


        # keep only African-American and Caucasian
        df = df.loc[df['race'].isin(['African-American','Caucasian']),:]
        
        # binarize race 
        # African-American: 0, Caucasian: 1
        df.loc[:,'race'] = df['race'].apply(lambda x: 1 if x=='Caucasian' else 0)
        
        # binarize gender
        # Female: 1, Male: 0
        df.loc[:,'sex'] = df['sex'].apply(lambda x: 1 if x=='Male' else 0)
        
        # binarize degree charged
        # Misd. = -1, Felony = 1
        df.loc[:,'c_charge_degree'] = df['c_charge_degree'].apply(lambda x: 1 if x=='F' else -1)
               
        num_train = df.shape[0]
        y = np.zeros(num_train)
        y[df['is_recid'].values == 1] = 1


        num_var_names = ['age',  'priors_count', 'length_of_stay'] 
        cat_var_names = ['race',  'sex', 'c_charge_degree']
            
        feature_names = num_var_names + cat_var_names
        df = df[feature_names]

        if verbose:
            print('Selecting relevant features complete.')

        for col in df:
            if col in cat_var_names:
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        dtypes = df.dtypes

        X = get_numpy(df)
        if verbose:
            print('Numpy conversion complete.')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # sens idx
        race_idx = 3
        sex_idx = 4
        sens_idc = [race_idx]

        race_cats = df[feature_names[race_idx]].cat.categories
        sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')


    elif name == 'hsls': 
        file ='../../data/HSLS/hsls_df.csv'
        df =pd.read_csv(file, index_col=0)

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
        df['race'] = np.logical_or(((df['studentrace']*7).astype(int)==7).values, ((df['studentrace']*7).astype(int)==1).values).astype(int)
        df['sex'] = df['studentgender'].astype(int)

        ## Dropping race and 12th grade data just to focus on the 9th grade prediction ##
        df = df.drop(columns=['studentgender', 'grade9thbin', 'grade12thbin', 'studentrace'])

        ## Scaling ##
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
        
        num_train = df.shape[0]
        y = np.zeros(num_train)
        y[df['gradebin'].values == 1] = 1
        df = df.drop(columns=['gradebin'])

        cat_var_names = ['race',  'sex']
        num_var_names = [x for x in df.columns.tolist() if x not in cat_var_names]
        feature_names = num_var_names + cat_var_names
        df = df[feature_names]

        if verbose:
            print('Selecting relevant features complete.')

        for col in df:
            if col in cat_var_names:
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        dtypes = df.dtypes

        X = get_numpy(df)
        if verbose:
            print('Numpy conversion complete.')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # sens idx
        race_idx = feature_names.index('race')
        sex_idx = feature_names.index('sex')
        sens_idc = [race_idx]

        race_cats = df[feature_names[race_idx]].cat.categories
        sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')


    elif name == 'enem':
        file ='../dataset/enem-50000-20.csv'
        df =pd.read_csv(file, index_col=0)

        df = df.rename(columns={'racebin': 'race', 'sexbin': 'sex'})

        num_train = df.shape[0]
        y = np.zeros(num_train)
        y[df['gradebin'].values == 1] = 1
        df = df.drop(columns=['gradebin'])

        col_names = df.columns.tolist()
        num_var_names = ['Q005']
        cat_var_names = [x for x in col_names if x not in num_var_names]
        feature_names = cat_var_names + num_var_names
        df = df[feature_names]

        if verbose:
            print('Selecting relevant features complete.')

        for col in df:
            if col in cat_var_names:
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        dtypes = df.dtypes

        X = get_numpy(df)
        if verbose:
            print('Numpy conversion complete.')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # sens idx
        race_idx = feature_names.index('race')
        sex_idx = feature_names.index('sex')
        sens_idc = [race_idx]

        race_cats = df[feature_names[race_idx]].cat.categories
        sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')


    elif name == 'hmda':
        # Read raw data from csv
        df_raw = pd.read_csv('../dataset/hmda_2017_all.csv')
        print('Raw Dataset loaded.')

        # Extract useful features and separate them according to num/cat vars
        feature_names = ['loan_type_name','property_type_name', 'loan_purpose_name','owner_occupancy_name','loan_amount_000s', \
                         'preapproval_name', 'msamd_name', 'state_name', \
                         'county_name', 'applicant_race_name_1', 'applicant_sex_name', \
                         'applicant_income_000s', 'purchaser_type_name', 'lien_status_name', \
                         'population', 'minority_population', 'hud_median_family_income', 'tract_to_msamd_income', \
                         'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']
        label_name = 'action_taken_name'
        num_train = df_raw.shape[0]
        y_raw = df_raw[label_name].values
        num_var_names = [x for x in feature_names if df_raw[x].dtypes == 'float64']
        cat_var_names = [x for x in feature_names if df_raw[x].dtypes != 'float64']
        feature_names = num_var_names + cat_var_names
        print(len(num_var_names), len(cat_var_names))
        df = df_raw[num_var_names + cat_var_names]
        print(num_var_names + cat_var_names)
        print('Selecting relevant features complete.')

        # Define what the labels are
        pos_labels = ['Loan originated', 'Loan purchased by the institution']
        neg_labels = ['Application approved but not accepted',
                      'Application denied by financial institution',
                      'Preapproval request denied by financial institution',
                      'Preapproval request approved but not accepted']

        pos_idx = np.array([])
        for l in pos_labels:
            pos_idx = np.concatenate((np.where(y_raw == l)[0], pos_idx))

        neg_idx = np.array([])
        for l in neg_labels:
            neg_idx = np.concatenate((np.where(y_raw == l)[0], neg_idx))

        X_pos = df.iloc[pos_idx][feature_names]
        X_neg = df.iloc[neg_idx][feature_names]
        # Remove rows with nan vals
        X_pos = X_pos.dropna(axis='rows')
        X_neg = X_neg.dropna(axis='rows')

        # Concat pos and neg samples
        Xdf = pd.concat((X_pos, X_neg))

        # Create labels
        y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
        for col in Xdf:
            if Xdf[col].dtype == 'object':
                Xdf[col] = Xdf[col].astype('category')
            else:
                Xdf[col] = Xdf[col].astype(np.float64)

        dtypes = Xdf.dtypes

        # get numpy format
        X = get_numpy(Xdf)

        print('Numpy conversion complete.')

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        print('Dataset split complete.')

        race_idx = 16
        sex_idx = 17

        race_cats = Xdf[feature_names[race_idx]].cat.categories
        sex_cats = Xdf[feature_names[sex_idx]].cat.categories

        #sensitive features
        sens_idc = [race_idx, sex_idx]

        # refer to these categorical tables for setting the index for positive and negative groups w.r.t senstive attr.
        print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        print('Sensitive attribute removal complete.')

    elif name == 'synth':
        def sample_from_gaussian(pos_mean,
                                 pos_cov,
                                 neg_mean,
                                 neg_cov,
                                 angle=np.pi/3,
                                 n_pos=200,
                                 n_neg=200,
                                 seed=0,
                                 corr_sens=True):
            np.random.seed(seed)
            x_pos = np.random.multivariate_normal(pos_mean, pos_cov, n_pos)
            np.random.seed(seed)
            x_neg = np.random.multivariate_normal(neg_mean, neg_cov, n_neg)
            X = np.vstack((x_pos, x_neg))
            y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
            n = y.shape[0]
            if corr_sens:
                # correlated sens data
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                xp = np.dot(X, rot)
                p1 = scp.stats.multivariate_normal.pdf(xp, pos_mean, pos_cov)
                p0 = scp.stats.multivariate_normal.pdf(xp, neg_mean, neg_cov)
                p = p1 / (p1 + p0)
                np.random.seed(seed)
                sens_attr = scp.stats.bernoulli.rvs(p)
            else:
                # independent sens data
                np.random.seed(seed)
                sens_attr = np.random.binomial(1, 0.5, n)
            return X, y, sens_attr

        ## NOTE change these variables for different distribution/generation of synth data.
        pos_mean = np.array([2,2])
        pos_cov = np.array([[5, 1], [1,5]])
        neg_mean = np.array([-2,-2])
        neg_cov = np.array([[10, 1],[1, 3]])
        n_pos = 500
        n_neg = 300
        angle = np.pi / 2
        #corr_sens = False
        X, y, sens = sample_from_gaussian(pos_mean,
                                          pos_cov,
                                          neg_mean,
                                          neg_cov,
                                          angle=angle,
                                          n_pos=n_pos,
                                          n_neg=n_neg,
                                          corr_sens=corr_sens)
        X = np.concatenate((X, np.expand_dims(sens, 1)), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        dtypes = None
        dtypes_ = None
        sens_idc = [2]
        X_train_removed = X_train[:,:2]
        X_test_removed = X_test[:,:2]
        race_idx = None
        sex_idx = None

    elif name == 'synth2':
        def sample_from_gaussian(pos_mean,
                                 pos_cov,
                                 neg_mean,
                                 neg_cov,
                                 thr=0,
                                 n_pos=200,
                                 n_neg=200,
                                 seed=0,
                                 corr_sens=True):
            np.random.seed(seed)
            x_pos = np.random.multivariate_normal(pos_mean, pos_cov, n_pos)
            np.random.seed(seed)
            x_neg = np.random.multivariate_normal(neg_mean, neg_cov, n_neg)
            X = np.vstack((x_pos, x_neg))
            y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
            n = y.shape[0]
            if corr_sens:
                # correlated sens data
                sens_attr = np.zeros(n)
                idx = np.where(X[:,0] > thr)[0]
                sens_attr[idx] = 1
            else:
                # independent sens data
                np.random.seed(seed)
                sens_attr = np.random.binomial(1, 0.5, n)
            return X, y, sens_attr

        ## NOTE change these variables for different distribution/generation of synth data.
        pos_mean = np.array([2,2])
        pos_cov = np.array([[5, 1], [1,5]])
        neg_mean = np.array([-2,-2])
        neg_cov = np.array([[10, 1],[1, 3]])
        n_pos = 500
        n_neg = 300
        thr = 0
        #corr_sens = False
        X, y, sens = sample_from_gaussian(pos_mean,
                                          pos_cov,
                                          neg_mean,
                                          neg_cov,
                                          thr=thr,
                                          n_pos=n_pos,
                                          n_neg=n_neg,
                                          corr_sens=corr_sens)
        X = np.concatenate((X, np.expand_dims(sens, 1)), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        dtypes = None
        dtypes_ = None
        sens_idc = [2]
        X_train_removed = X_train[:,:2]
        X_test_removed = X_test[:,:2]
        race_idx = None
        sex_idx = None

    elif name == 'german':
        # Download data if needed
        _german_loan_attribute_map = dict(
            A11='< 0 DM',
            A12='0-200 DM',
            A13='>= 200 DM',
            A14='no checking',
            A30='no credits',
            A31='all credits paid back',
            A32='existing credits paid back',
            A33='delayed past payments',
            A34='critical account',
            A40='car (new)',
            A41='car (used)',
            A42='furniture/equipment',
            A43='radio/television',
            A44='domestic appliances',
            A45='repairs',
            A46='education',
            A47='(vacation?)',
            A48='retraining',
            A49='business',
            A410='others',
            A61='< 100 DM',
            A62='100-500 DM',
            A63='500-1000 DM',
            A64='>= 1000 DM',
            A65='unknown/no sav acct',
            A71='unemployed',
            A72='< 1 year',
            A73='1-4 years',
            A74='4-7 years',
            A75='>= 7 years',
            #A91='male & divorced',
            #A92='female & divorced/married',
            #A93='male & single',
            #A94='male & married',
            #A95='female & single',
            A91='male',
            A92='female',
            A93='male',
            A94='male',
            A95='female',
            A101='none',
            A102='co-applicant',
            A103='guarantor',
            A121='real estate',
            A122='life insurance',
            A123='car or other',
            A124='unknown/no property',
            A141='bank',
            A142='stores',
            A143='none',
            A151='rent',
            A152='own',
            A153='for free',
            A171='unskilled & non-resident',
            A172='unskilled & resident',
            A173='skilled employee',
            A174='management/self-employed',
            A191='no telephone',
            A192='has telephone',
            A201='foreigner',
            A202='non-foreigner',
        )

        filename = 'german.data'
        if not os.path.isfile(filename):
            print('Downloading data to %s' % os.path.abspath(filename))
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                                       filename)

        # Load data and setup dtypes
        col_names = [
            'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
            'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
            'other_debtors', 'residing_since', 'property', 'age',
            'inst_plans', 'housing', 'num_credits',
            'job', 'dependents', 'telephone', 'foreign_worker', 'status']
        df = pd.read_csv(filename, delimiter=' ', header=None, names=col_names)
        for k, v in _german_loan_attribute_map.items():
            df.replace(k, v, inplace=True)
        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        def get_numpy(df):
            new_df = df.copy()
            cat_columns = new_df.select_dtypes(['category']).columns
            new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
            return new_df.values
        Xy = get_numpy(df)
        X = Xy[:,:-1]
        y = Xy[:,-1]
        # Make 1 (good customer) and 0 (bad customer)
        # (Originally 2 is bad customer and 1 is good customer)
        sel_bad = y == 2
        y[sel_bad] = 0
        y[~sel_bad] = 1
        feature_labels = df.columns.values[:-1]  # Last is prediction
        dtypes = df.dtypes[:-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        # Senstivie attribute
        foreign = 19
        age = 12
        sex_idx = 8
        sens_idc = [sex_idx, age, foreign]

        foreign_cats = df[feature_labels[foreign]].cat.categories
        sex_cats = df[feature_labels[sex_idx]].cat.categories
        print(foreign_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        race_idx = foreign

    else:
        raise ValueError('Data name invalid.')

    return X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, dtypes, dtypes_, sens_idc, race_idx, sex_idx

def get_csv_eqodd(fm, data_name='adult'):
    assert(fm.model is not None)
    # get csv files required for the eq_odds code to run
    label = fm.y_test
    group = fm.X_test[:, fm.sens_idx]
    prediction = fm.model.predict_proba(fm.X_test_removed)[:,1] # positive label prediction
    # make csv file
    f = open('%s_predictions.csv'%data_name, 'w')
    f.write(',label,group,prediction\n')
    for i, e in enumerate(zip(label, group, prediction)):
        line = '%d,%0.2f,%0.2f,%f\n'%(i, e[0],  e[1], e[2])
        f.write(line)
    f.close()


    
