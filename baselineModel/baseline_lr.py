import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix, vstack
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error, log_loss, accuracy




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
    
    train_ix, test_ix = splitter(train)

    train_lr = train.ix[train_ix].copy()
    test_lr = train.ix[test_ix].copy()

    #Only keep the necessary features for the baseline model
    train_lr = remove_unused_columns(train_lr)
    test_lr = remove_unused_columns(test_lr)

    reset_index(train_lr)
    reset_index(test_lr)


    #Train set
    #Create sparse X_train matrix
    X_train_steps = create_sparse_occurences(train_lr, 'step_id')
    X_train_stud = create_sparse_occurences(train_lr, 'student_id')
    X_train = hstack((X_train_stud, X_train_steps))

    y_train = train_lr.y_one_negative_one
    y01_train = train_lr.correct_first_attempt

    #Test set
    X_test_steps = create_sparse_occurences(test_lr, 'step_id')
    X_test_stud = create_sparse_occurences(test_lr, 'student_id')
    X_test = hstack((X_test_stud, X_test_steps))    

    y_test = train_lr.y_one_negative_one
    y01_test = train_lr.correct_first_attempt
    

    #Grid of N for regularization in cross validation
    N = 15
    Cs = np.linspace(1e-4, 1e2, N)
    lr = LogisticRegressionCV(Cs = Cs, fit_intercept=True, penalty='l2', 
        scoring='log_loss', n_jobs=3)
    lr.fit(X_train, y_train)

    #Evaluation in train set
    pred_proba_train = lr.predict_proba(X_train)
    pred_proba_train_1 = [x[1] for x in pred_proba_train]
    pred_class_train = lr.predict(X_train)

    mse_train = mean_squared_error(y_01_train, pred_proba_train_1)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y_01_train, pred_proba_train_1)


    #Evaluation in test set
    pred_proba_test = lr.predict_proba(X_test)
    pred_proba_test_1 = [x[1] for x in pred_proba_test]
    pred_class_test = lr.predict(X_test)

    mse_test = mean_squared_error(y_01_test, pred_proba_test_1)
    rmse_test = np.sqrt(mse_test)
    logloss_test = log_loss(y_01_test, pred_proba_test_1)


if __name__ == '__main__':
    main()
