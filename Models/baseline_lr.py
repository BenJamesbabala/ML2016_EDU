import numpy as np
import pandas as pd

from scipy.sparse import hstack, coo_matrix, vstack, csr_matrix
import scipy

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score




def remove_unused_columns(ds):

    unused_columns = [u'problem_name', u'view', u'step_name',
       u'start_time', u'first_trans_time', u'correct_trans_time', u'end_time',
       u'step_duration', u'correct_step_duration', u'error_step_duration',
       u'incorrects', u'hints', u'corrects',
       u'kc_subskills', u'opp_subskills', u'k_traced_skills', u'opp_k_traced',
       u'kc_rules', u'opp_rules', u'unit', u'section', u'problem_id']

    return ds.drop(unused_columns,1)


def create_sparse_occurences(ds, column):
    grouped = ds.groupby(column)
    groups = grouped.groups
    indices = groups.values()

    lengths = []
    for i,index in enumerate(indices):
        lengths.append([i]*len(index))

    inds = np.array([item for sublist in indices for item in sublist])
    cols = np.array([item for sublist in lengths for item in sublist])
    values = np.ones(len(cols))

    return scipy.sparse.csr_matrix((values, (inds, cols)))


def create_dummy_representation(ds):
    cols = [u'student_id', u'step_id']
    #cols = [u'student_id']
    return pd.get_dummies(ds, columns = cols, sparse=True)



def main():
    
    ds_lr = remove_unused_columns(train)
    X_step = create_sparse_occurences(ds_lr, 'step_id')
    X_stud = create_sparse_occurences(ds_lr, 'student_id')
    
    X = csr_matrix(hstack((X_stud, X_step)))

    train_ix, test_ix = splitter(train)

    X_train = X[train_ix]
    X_test = X[test_ix]

    
    train_lr = ds_lr.ix[train_ix]
    y_train = train_lr.y_one_negative_one
    y01_train = train_lr.correct_first_attempt

    test_lr = ds_lr.ix[test_ix]
    y_test = test_lr.y_one_negative_one
    y01_test = test_lr.correct_first_attempt


    #Grid of N for regularization in cross validation
    N = 10
    Cs = np.logspace(-6, 2, num=N)
    lr = LogisticRegressionCV(Cs = Cs, fit_intercept=True, penalty='l2', 
        scoring='log_loss', n_jobs=4)


    lr.fit(X_train, y_train)

    #Evaluation in train set
    pred_proba_train = lr.predict_proba(X_train)
    pred_proba_train_1 = [x[1] for x in pred_proba_train]
    pred_class_train = lr.predict(X_train)

    mse_train = mean_squared_error(y01_train, pred_proba_train_1)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y01_train, pred_proba_train_1)
    accuracy_train = accuracy_score(y_train,pred_class_train)


    #Evaluation in test set
    pred_proba_test = lr.predict_proba(X_test)
    pred_proba_test_1 = [x[1] for x in pred_proba_test]
    pred_class_test = lr.predict(X_test)

    mse_test = mean_squared_error(y01_test, pred_proba_test_1)
    rmse_test = np.sqrt(mse_test)
    logloss_test = log_loss(y01_test, pred_proba_test_1)
    accuracy_test= accuracy_score(y_test,pred_class_test)


if __name__ == '__main__':
    main()
