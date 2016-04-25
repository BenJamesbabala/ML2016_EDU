import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix, vstack
import scipy
from sklearn.linear_model import LinearRegression




def remove_unused_columns(ds):

    unused_columns = [u'problem_name', u'view', u'step_name',
       u'start_time', u'first_trans_time', u'correct_trans_time', u'end_time',
       u'step_duration', u'correct_step_duration', u'error_step_duration',
       u'correct_first_attempt', u'incorrects', u'hints', u'corrects',
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

    train_lr = remove_unused_columns(train)
    train_lr_d = create_dummy_representation(train_lr)

    #sub_students = np.random.choice(students, 200)

#    students = train_lr.student_id.unique()
#    nstudents = len(students)
#    steps = train_lr.step_id.unique()
#    nsteps = len(steps)
#    ninstances = train_lr.shape[0]
    X_steps = create_sparse_occurences(train_lr, 'step_id')
    X_stud = create_sparse_occurences(train_lr, 'student_id')
    X = hstack((X_stud, X_steps))
    y = train_lr.y_one_negative_one

    lr = LinearRegression()







        instance = train_lr.ix[index]
        student = instance.student_id
        step = instance.step_id

        student_ix = np.where(students == student)[0][0]
        step_ix = nstudents + np.where(steps == step)[0][0]
        
        tmp = scipy.sparse.csr_matrix((1,nstudents+nsteps), dtype=np.int8)
        tmp[0, student_ix] = 1
        tmp[0, step_ix] = 1

        if row%10000 == 0:
            print row

        dummies = vstack([dummies,tmp])






def encoding_typestr(a):
    return str(type(a)) == "<type 'str'>"



if __name__ == '__main__':
    main()
