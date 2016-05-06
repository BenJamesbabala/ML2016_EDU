import numpy as np
import pandas as pd

from scipy.sparse import hstack, coo_matrix, vstack, csr_matrix, lil_matrix
import scipy

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score

import time

def get_non_numeric_features(ds):
    ''' Return a list with the names of the columns
    in the dataFrame which are not numeric ''' 
    types = ds.dtypes
    types = dict(types)
    numeric_types = [np.int32, np.int64, np.float64]
    non_numeric_features = []
    
    for col, ty in types.items():
        if ty not in numeric_types:
            non_numeric_features.append(col)

    return non_numeric_features


def concat_sparse_w_df(sparse, ds):
    ''' Receives a sparse matrix and a DataFrame
    of the same size of rows and returns its concatenation
    in sparse matrix format '''

    X_ds_matrix = ds.as_matrix()
    X_ds_sparse = csr_matrix(X_ds)

    X_sparse = hstack([X_ds_sparse, sparse])
    
    return csr_matrix(X_sparse)




def main():


    #Remove non-numeric features
    non_num_features = get_non_numeric_features(ds)
    ds_num = ds.drop(non_num_features, 1)

    X_ds = ds_num.drop(['correct_first_attempt', 'y_one_negative_one'], 1)

    not_used_columns = [u'step_duration', u'correct_step_duration', u'error_step_duration',
                         u'row']

    X_ds = ds_num.drop(not_used_columns, 1)

    #Concat sparse matrix and dataframe
    X = concat_sparse_w_df(skills_sparse_cl, X_ds)

    #Split X in train and test
    X_train = X[train_ix]
    X_test = X[test_ix]

    y = ds_num.correct_first_attempt
    y_train = y.loc[train_ix]
    y_test = y.loc[test_ix]


    #Grid of N for regularization in cross validation
    N = 5
    Cs = np.logspace(-6, 2, num=N)
    lr = LogisticRegressionCV(Cs = Cs, fit_intercept=True, penalty='l2', 
        scoring='log_loss', n_jobs=6)

    print time.ctime()
    lr.fit(X_train, y_train)
    print time.ctime()


    #Evaluation in train set
    pred_proba_train = lr.predict_proba(X_train)
    pred_proba_train_1 = [x[1] for x in pred_proba_train]
    pred_class_train = lr.predict(X_train)

    mse_train = mean_squared_error(y_train, pred_proba_train_1)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y_train, pred_proba_train_1)
    accuracy_train = accuracy_score(y_train,pred_class_train)


    #Evaluation in test set
    pred_proba_test = lr.predict_proba(X_test)
    pred_proba_test_1 = [x[1] for x in pred_proba_test]
    pred_class_test = lr.predict(X_test)

    mse_test = mean_squared_error(y_test, pred_proba_test_1)
    rmse_test = np.sqrt(mse_test)
    logloss_test = log_loss(y_test, pred_proba_test_1)
    accuracy_test= accuracy_score(y_test,pred_class_test)



if __name__ == '__main__':
    main()