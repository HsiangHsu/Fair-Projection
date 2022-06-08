import pandas as pd
import numpy as np
import scipy as scp
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

def is_categorical(dtypes):
    # Copied from counterfactual.funcs module for convenience
    def check_dtype(dtype):
        if pd.api.types.is_categorical_dtype(dtype):
            return True
        try:
            dtype.categories
        except AttributeError:
            try:
                dtype['categories']
            except TypeError:
                return False
            except KeyError:
                return False
            else:
                return True
        else:
            return True
    return np.array([check_dtype(dtype) for dtype in dtypes])

def _create_pipe(estimator, dtypes, sparse=False, with_mean=True):
    return make_pipeline(
        OneHotEncoder(sparse=sparse, handle_unknown='ignore'),
        # OneHotEncoder(sparse=sparse, categories=is_categorical(dtypes)),
        #n_values=np.array([len(dtype.categories) for dtype in dtypes if hasattr(dtype, 'categories')])),
        StandardScaler(with_mean=with_mean),
        estimator,
    )

def _create_cv_pipe(estimator, param_grid, dtypes, random_state=0, sparse=False, with_mean=True):
    pipe = _create_pipe(estimator, dtypes, sparse=sparse, with_mean=with_mean)
    cv = StratifiedKFold(5, random_state=random_state)
    return GridSearchCV(pipe, param_grid, scoring='accuracy', iid=False, cv=cv, refit=True)

def create_and_train_models(model_names, dtypes, X_train, y_train, X_test=None, y_test=None, save=False, data_name='adult'):
    model_dicts = np.array([
        create_model(model_name, dtypes, random_state=0)
        for model_name in model_names
    ])
    for d in model_dicts:
        print('Training %s'%d)
        try:
            d['estimator'].fit(X_train, y_train)
        except MemoryError:
            print('Memory Error: Try using BatchEstimators')
            assert(False)
        d['train_score'] = accuracy_score(d['estimator'].predict(X_train), y_train)
        if X_test is not None and y_test is not None:
            d['test_score'] = accuracy_score(d['estimator'].predict(X_test), y_test)
        d['model'] = d['get_model'](d['estimator'])
    if save:
        pickle.dump([d['estimator'] for d in model_dicts], open('model_%s.pkl'%data_name, 'wb'))
    return model_dicts

def create_model(model_name, dtypes, random_state=42):
    if model_name == 'RBFSVM':
        base_estimator = SVC(random_state=random_state)
        param_grid = {
            'svc__C': np.logspace(-1, 3, 10),
            'svc__gamma': np.logspace(-4, 1, 10),
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    elif model_name == 'LogisticRegression':
        base_estimator = LogisticRegression(random_state=42)
        param_grid = {
            'logisticregression__C': np.logspace(-4, -1, 2),
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    elif model_name == 'DecisionTree':
        base_estimator = DecisionTreeClassifier(random_state=random_state)
        param_grid = {
            'decisiontreeclassifier__max_leaf_nodes': [5, 10, 20, 40],
            'decisiontreeclassifier__max_depth': np.arange(10) + 1,
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def decision_tree_model(X):
                func = estimator.predict_proba
                if np.asarray(X).ndim == 1:
                    p = func(X.reshape(1, -1))[0][1]
                else:
                    p = func(X)[:, 1]
                return _logit(p)
            return decision_tree_model
    elif model_name == 'GradientBoost':
        base_estimator = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
            #'gradientboostingclassifier__max_depth': [1, 2, 3, 4, 5],
            'gradientboostingclassifier__n_estimators': [25, 50, 100, 200, 500],#[1, 2, 3, 4, 5],
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def gradient_boost_model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return gradient_boost_model
    elif model_name == 'RandomForest':
        base_estimator = RandomForestClassifier(n_estimators=10, min_samples_leaf=10, random_state=random_state)
        param_grid = {
            'randomforestclassifier__max_depth': [4, 5, 6],
            'randomforestclassifier__n_estimators': [100, 200, 400],#[1, 2, 3, 4, 5],
        }

        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
        def get_model(estimator):
            def random_forest_model(X):
                func = estimator.predict_proba
                if np.asarray(X).ndim == 1:
                    p = func(X.reshape(1, -1))[0][1]
                else:
                    p = func(X)[:, 1]
                return _logit(p)
            return random_forest_model
    elif model_name == 'BatchSGD_LogReg':
        base_estimator = BatchEstimator(loss='log', random_state=random_state)
        param_grid = {
            'batchestimator__alpha': [0.0001, 0.0005]
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state, sparse=True, with_mean=False)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    elif model_name == 'BatchSGD_SVM':
        base_estimator = BatchEstimator(loss='hinge', random_state=random_state)
        param_grid = {
            'batchestimator__alpha': [0.0001, 0.0005]
        }
        estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state, sparse=True, with_mean=False)
        def get_model(estimator):
            def model(X):
                func = estimator.decision_function
                if np.asarray(X).ndim == 1:
                    return func(X.reshape(1, -1))[0]
                return func(X)
            return model
    # elif model_name == 'DNN':
    #     base_estimator = _DNNEstimator(max_epoch=1000, lr=1e-4, batch=1000, random_state=random_state)
    #     param_grid = {
    #         '_dnnestimator__batch': [100, 200, 400],
    #     }
    #     estimator = _create_cv_pipe(base_estimator, param_grid, dtypes, random_state=random_state)
    #     def get_model(estimator):
    #         def dnn_model(X):
    #             func = estimator.decision_function
    #             if np.asarray(X).ndim == 1:
    #                 p = func(X.reshape(1, -1))[0]
    #             else:
    #                 p = func(X)
    #             return _logit(p)
    #         return dnn_model
    else:
        raise ValueError('Could not recognize "%s" model name.' % model_name)

    return dict(estimator=estimator, get_model=get_model, model_name=model_name)

class BatchEstimator(BaseEstimator):
    def __init__(self, loss, random_state=0, batch_size=1000000, alpha=0.00001):
        self.loss = loss
        self.batch_size = batch_size
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        model = SGDClassifier(loss=self.loss, alpha=self.alpha, random_state=self.random_state)
        dataiter = get_batches(X, y, batch_size=self.batch_size)
        for xb, yb, bs in dataiter:
            print('%d / %d'%(bs, X.shape[0]))
            model.partial_fit(xb, yb, np.unique(y))
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError('probabilistic output not supported')


class ConstEstimator(BaseEstimator):
    def __init__(self, pos=True):
        self.pos = pos

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if self.pos:
            return np.array([1] * X.shape[0])
        else:
            return np.array([0] * X.shape[0])

class RandomEstimator(BaseEstimator):
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        np.random.seed(self.seed)
        return np.random.randint(2, size=X.shape[0])

def get_batches(X, y, batch_size=64):
    def get_rows(idx):
        return X[idx, :], y[idx]
    num_data = X.shape[0]
    batch_start = 0
    while batch_start < num_data:
        if batch_start + batch_size > num_data:
            batch_idx = range(batch_start, num_data)
        else:
            batch_idx = range(batch_start, batch_start+batch_size)
        X_batch, y_batch = get_rows(batch_idx)
        yield X_batch, y_batch, batch_start
        batch_start += batch_size

def _logit(p):
    p = np.minimum(1-1e-7, np.maximum(p, 1e-7))
    assert np.all(p < 1) and np.all(p > 0)
    return np.log(p/(1-p))
"""
#######################
# Deep neural network
#######################
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class _NN(nn.Module):
    def __init__(self, input_dim, num_class):
        super(_NN, self).__init__()
        self.input_dim = input_dim
        self.num_class = num_class
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def _nn_train(model, dataset, max_epoch=5000, lr=0.01, batch=64, data_name='german', seed=None):
    # create your optimizer
    if seed is not None:
        torch.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for n_epochs in range(max_epoch):
        loss_vals = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            optimizer.zero_grad()   # zero the gradient buffers
            input, target = sample_batched
            output = model(input.float())
            one_hot_targets = torch.from_numpy(np.eye(2)[target]).float()
            loss = criterion(output, one_hot_targets)
            loss.backward()
            optimizer.step()
            loss_vals += loss.item()
        if n_epochs % 100 == 0:
            print('ep%d\tloss:%f'%(n_epochs, loss_vals))

    torch.save(model.state_dict(), '%s_dnn.pt'%(data_name))

class _DNNEstimator(BaseEstimator):
    def __init__(self, max_epoch=5000, lr=0.01, batch=64, data_name='german', random_state=None):
        self.max_epoch = max_epoch
        self.lr = lr
        self.batch = batch
        self.data_name = data_name
        self.random_state = random_state

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        torch.manual_seed(rng.randint(2**32-1))

        self.classes_ = np.unique(y)
        model = _NN(n_features, len(self.classes_))
        dataset = _CreateDataset(X, y)
        _nn_train(model, dataset, max_epoch=self.max_epoch,
                  lr=self.lr, batch=self.batch, data_name=self.data_name)
        self.model_ = model
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.model_(torch.as_tensor(X).float()).detach().numpy(), axis=1)]

    def decision_function(self, X):
        return self.model_(torch.as_tensor(X).float()).detach().numpy()[:, 1]
"""

