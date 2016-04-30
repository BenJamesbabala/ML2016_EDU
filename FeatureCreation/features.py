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
    # SHIFT EVERYTHING BY ONE NOT TO LEAK!!!!!!!!
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




def skills_corr_counter_win(ds, sparse_matrix_input):

    #New Version: 
    # Timing:   Wed Apr 27 02:21:00 2016
    #           Wed Apr 27 02:25:35 2016
    student_cfa = ds[['student_id', 'correct_first_attempt']]
    sparse_matrix = csr_matrix(sparse_matrix_input.shape)

    for col in xrange(sparse_matrix_input.shape[1]):

        skill_indices = np.array(sparse_matrix_input[:,col].nonzero()[0])

        s_cfa = student_cfa.ix[skill_indices]
        grouped = s_cfa.groupby('student_id')
        sg = grouped.apply(cumsum_window)
        sg = sg.reset_index(level=0).drop('student_id',axis=1)

        
        indices = np.array(sg.index)
        values = sg.values

        sparse_matrix[indices,col] = values

    return sparse_matrix


6277028    stu_ffe1e527a8
6282419    stu_ffe1e527a8
6286301    stu_ffe1e527a8
6293695    stu_ffe1e527a8
6304565    stu_ffe1e527a8
6307077    stu_ffe1e527a8
6334143    stu_ffe1e527a8
6337164    stu_ffe1e527a8
6407736

[6277028, 
6282419,
6286301,
6293695,
6304565,
6307077,
6334143,
6337164,
6407736]



def cumsum_window(obs, N=5):
    cum = obs.cumsum().correct_first_attempt
    cum_delay = cum.shift(N).fillna(0)

    den = np.arange(len(cum))+1
    den[N:] = N

    diff = cum - cum_delay
    diff = diff/den
    diff = diff.shift(1).fillna(0)
    
    return diff

    



########################################
#   OLD VERSION
########################################3
# def skills_corr_counter(ds, sparse_matrix_input):

#     #This was the second version. The part that is still too slow is the filtering and groupby of the ds
#     students = ds.student_id.unique()
#     student_cfa = ds[['student_id', 'correct_first_attempt']]
#     students_group = student_cfa.groupby('student_id')
#     groups = students_group.groups
#     sparse_matrix = csr_matrix(sparse_matrix_input.shape)

#     for col in xrange(sparse_matrix_input.shape[1]):

#         #new_column = pd.Series(np.zeros(ds.shape[0]))

#         skill_indices = np.array(sparse_matrix_input[:,col].nonzero()[0])

#         for student in students:
#             student_index = np.array(groups[student])
#             indices = np.intersect1d(skill_indices, student_index, assume_unique=True)
            
#             cumsum = np.array(student_cfa.ix[indices].correct_first_attempt.cumsum())
#             cumsum = cumsum.reshape(len(cumsum),1)
#             sparse_matrix[indices,col] = cumsum


#             #new_column.loc[indexes] = np.array(student_cfa.correct_first_attempt.cumsum())
#             #new_column.loc[indexes] = ds.ix[indexes].groupby('student_id').cumsum().correct_first_attempt.reshape((indexes.shape[0], 1))
#     return sparse_matrix

print time.ctime()
subskills_count_win = skills_corr_counter_win(ds, subskills_sparse)
print time.ctime()