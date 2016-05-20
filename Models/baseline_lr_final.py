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

    idx = groups.keys()

    return scipy.sparse.csr_matrix((values, (inds, cols))), idx


def create_dummy_representation(ds):
    cols = [u'student_id', u'step_id']
    #cols = [u'student_id']
    return pd.get_dummies(ds, columns = cols, sparse=True)

def merge_estimates_w_data(data,beta, theta):
    

    merged = data.merge( beta, how = 'left',left_on='step_id',
                        right_on='step_id')
    merged = merged.merge(theta, how = 'left',
                            left_on='student_id',
                            right_on='student_id')
    
    merged.beta = merged.beta.astype(np.float64)
    merged.theta = merged.theta.astype(np.float64)
    return merged



def main():
    
    ########################
    # PREPARE DATASETS
    ########################

    ds_lr = remove_unused_columns(ds)
    X_step, step_idx = create_sparse_occurences(ds_lr, 'step_id')
    X_stud, stud_idx = create_sparse_occurences(ds_lr, 'student_id')
    
    X = csr_matrix(hstack((X_stud, X_step)))

    train_ix, test_ix = splitter(train)

    X_train = X[train_ix]
    X_val = X[val_ix]
    X_test = X[test_ix]

    
    train_lr = ds_lr.ix[train_ix]
    y_train = train_lr.y_one_negative_one
    y01_train = train_lr.correct_first_attempt
    
    val_lr = ds_lr.ix[val_ix]
    y_val = val_lr.y_one_negative_one
    y01_val = val_lr.correct_first_attempt

    test_lr = ds_lr.ix[test_ix]
    y_test = test_lr.y_one_negative_one
    y01_test = test_lr.correct_first_attempt


    ########################
    # GRID SEARCH
    ########################

    N = 5
    Cs = np.logspace(-4, 2, num=N)
    penalties = ['l1', 'l2']

    models = []
    train_ll = []
    val_ll = []
    train_rmse = []
    val_rmse = []

    for penalty in penalties:
        for C in Cs:
            lr = LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=C,
                                    fit_intercept=True, intercept_scaling=1, 
                                    class_weight=None, random_state=None, 
                                    solver='liblinear', max_iter=100, 
                                    multi_class='ovr', verbose=0, 
                                    warm_start=False, n_jobs=4)

            lr.fit(X_train, y_train)
            print penalty
            print C
            print 'Train Completed'

            #Evaluation in train set
            pred_proba_train = lr.predict_proba(X_train)
            pred_proba_train_1 = [x[1] for x in pred_proba_train]
        
            mse_train = mean_squared_error(y01_train, pred_proba_train_1)
            rmse_train = np.sqrt(mse_train)
            train_rmse.append(rmse_train)

            logloss_train = log_loss(y01_train, pred_proba_train_1)
            train_ll.append(logloss_train)

        
            #Evaluation in validation set
            pred_proba_val = lr.predict_proba(X_val)
            pred_proba_val_1 = [x[1] for x in pred_proba_val]

        
            mse_val = mean_squared_error(y01_val, pred_proba_val_1)
            rmse_val = np.sqrt(mse_val)
            val_rmse.append(rmse_val)

            logloss_val = log_loss(y01_val, pred_proba_val_1)
            val_ll.append(logloss_val)



    ########################
    # NORMAL TRAIN
    ########################
    penalty = 'l2'
    C=3.16
    lr = LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=C,
                        fit_intercept=True, intercept_scaling=1, 
                        class_weight=None, random_state=None, 
                        solver='liblinear', max_iter=100, 
                        multi_class='ovr', verbose=0, 
                        warm_start=False, n_jobs=4)
    lr.fit(X_train, y_train)

    #Evaluation in train set
    pred_proba_train = lr.predict_proba(X_train)
    pred_proba_train_1 = [x[1] for x in pred_proba_train]

    mse_train = mean_squared_error(y01_train, pred_proba_train_1)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y01_train, pred_proba_train_1)
    
    #Evaluation in val set
    pred_proba_val = lr.predict_proba(X_val)
    pred_proba_val_1 = [x[1] for x in pred_proba_val]
    
    mse_val = mean_squared_error(y01_val, pred_proba_val_1)
    rmse_val = np.sqrt(mse_val)
    logloss_val = log_loss(y01_val, pred_proba_val_1)
    
    #Evaluation in test set
    pred_proba_test = lr.predict_proba(X_test)
    pred_proba_test_1 = [x[1] for x in pred_proba_test]
    
    mse_test = mean_squared_error(y01_test, pred_proba_test_1)
    rmse_test = np.sqrt(mse_test)
    logloss_test = log_loss(y01_test, pred_proba_test_1)

rmse_train
rmse_test
logloss_train
logloss_test

    #CREATE THETA AND BETA DATAFRAMES
    #USED IN INIT.PY
    # beta = pd.DataFrame(np.array([beta_prob, step_idx]).T, 
    #                     columns=['beta','step_id'])
    # theta = pd.DataFrame(np.array([theta_stud, stud_idx]).T, 
    #                     columns=['theta','student_id'])







if __name__ == '__main__':
    main()


