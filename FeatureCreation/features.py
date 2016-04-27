import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix, vstack, csr_matrix, lil_matrix
import scipy
import sklearn

def skills_knowledge_counter(df, features, sparse_matrix_input, subskills_col):

    #This was the first version of the function
    students = df.student_id.unique()


    for skill in features:

        new_column = pd.DataFrame(np.zeros(df.shape[0]))
        for student in students:
            index = df[(df[subskills_col].str.contains(skill)) & (df.student_id == student)].index
            new_column.loc[index] = df.loc[index].groupby('student_id').cumsum().correct_first_attempt.reshape((index.shape[0], 1))

        sparse_col = coo_matrix(new_column)

        if sparse_matrix:
            sparse_matrix = hstack([sparse_matrix, sparse_col])
        else:
            sparse_matrix = sparse_col


    return sparse_matrix



def skills_corr_counter(ds, sparse_matrix_input):

    #This was the second version. The part that is still too slow is the filtering and groupby of the ds
    students = ds.student_id.unique()
    student_cfa = ds[['student_id', 'correct_first_attempt']]
    students_group = student_cfa.groupby('student_id')
    groups = students_group.groups
    sparse_matrix = csr_matrix(sparse_matrix_input.shape)

    for col in xrange(sparse_matrix_input.shape[1]):

        #new_column = pd.Series(np.zeros(ds.shape[0]))

        skill_indices = np.array(sparse_matrix_input[:,col].nonzero()[0])

        for student in students:
            student_index = np.array(groups[student])
            indices = np.intersect1d(skill_indices, student_index, assume_unique=True)
            
            cumsum = np.array(student_cfa.ix[indices].correct_first_attempt.cumsum())
            cumsum = cumsum.reshape(len(cumsum),1)
            sparse_matrix[indices,col] = cumsum


            #new_column.loc[indexes] = np.array(student_cfa.correct_first_attempt.cumsum())
            #new_column.loc[indexes] = ds.ix[indexes].groupby('student_id').cumsum().correct_first_attempt.reshape((indexes.shape[0], 1))
    return sparse_matrix

time.ctime()
subskills_count = skills_corr_counter(ds, subskills_sparse)
time.ctime()



def skills_corr_counter(ds, sparse_matrix_input):

    #New Version: 
    # Timing:   Wed Apr 27 02:21:00 2016
    #           Wed Apr 27 02:25:35 2016

    student_cfa = ds[['student_id', 'correct_first_attempt']]
    sparse_matrix = csr_matrix(sparse_matrix_input.shape)

    for col in xrange(sparse_matrix_input.shape[1]):

        skill_indices = np.array(sparse_matrix_input[:,col].nonzero()[0])

        s_cfa = student_cfa.ix[skill_indices]
        sg = s_cfa.groupby('student_id').cumsum()
        indices = np.array(sg.index)
        values = sg.values

        sparse_matrix[indices,col] = values

    return sparse_matrix
