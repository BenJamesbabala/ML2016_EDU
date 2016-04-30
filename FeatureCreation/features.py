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



def skills_corr_counter_win(ds, sparse_matrix_input, window=None):
    #If window not specified not use window
    student_cfa = ds[['student_id', 'correct_first_attempt']]
    sparse_matrix = csr_matrix(sparse_matrix_input.shape)

    for col in xrange(sparse_matrix_input.shape[1]):

        skill_indices = np.array(sparse_matrix_input[:,col].nonzero()[0])

        s_cfa = student_cfa.ix[skill_indices]
        grouped = s_cfa.groupby('student_id')

        if window:
            sg = grouped.apply(cumsum_window, window)
            sg = sg.reset_index(level=0).drop('student_id',axis=1)
        else:
            sg = grouped.cumsum()
        
        indices = np.array(sg.index)
        values = sg.values

        sparse_matrix[indices,col] = values

    return sparse_matrix

def cumsum_window(obs, N=5):
    cum = obs.cumsum().correct_first_attempt
    cum_delay = cum.shift(N).fillna(0)

    den = np.arange(len(cum))+1
    den[N:] = N

    diff = cum - cum_delay
    diff = diff/den
    diff = diff.shift(1).fillna(0)
    
    return diff


def create_and_save_sparses(ds, sparse_list, window_list):

    sp_n = 0

    for sp in sparse_list:
        sp_n = sp_n+1
        for window in window_list:
            sparse_cum = skills_corr_counter_win(ds,
                                sp, window=window)

            np.save('matrix_'+str(sp_n)+'_win_'+str(window)+'.npy',
                    sparse_cum) 
        print ('Matrix number {} window {} completed'.format(sp_n, window))


def previous_correct_first_attempt_column(data_frame):
    #data_frame: pandas dataframe

    ds_grouped = data_frame.groupby(['student_id', 'step_id'])
    indices = ds_grouped.indices
    data_frame['correct_first_attempt_previous'] = -0.5

    for student, step_id in indices:

            indexes_student_step = indices[student, step_id]


            data_step_student = data_frame.ix[indices[student, step_id]]
            correct_first_attempt_delayed = data_step_student['correct_first_attempt'].shift(1)
            data_frame.loc[(indices[student, step_id], 'correct_first_attempt_previous')] = correct_first_attempt_delayed

    data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous == 0] = -1
    data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous == -0.5] = 0
    data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous.isnull()] = 0

#sparse_list = [subskills_sparse, k_traced_sparse, kc_rules_sparse]
#windows = [1,2,3,4,5,6,7,8,9,10]
# create_and_save_sparses(train, sparse_list, windows)





def previous_correct_first_attempt_column(data_frame):
    #data_frame: pandas dataframe
    data_frame = data_frame[['student_id', 'step_id',
                            'correct_first_attempt']]

    ds_grouped = data_frame.groupby(['student_id', 'step_id'])
      
    #cfa_prev = pd.DataFrame(np.ones(data_frame.shape[0])*(-0.5), 
    #                        index=data_frame.index, columns=['cfa_prev'])

    #data_frame['correct_first_attempt_previous'] = -0.5

    cfa_prev = ds_grouped.apply(previous_cfa)
    
    #data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous == 0] = -1
    #data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous.isnull()] = 0

def previous_cfa(df):
    cfa_prev = df.correct_first_attempt.shift(1)
    return cfa_prev




    data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous == 0] = -1
    data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous == -0.5] = 0
    data_frame.correct_first_attempt_previous[data_frame.correct_first_attempt_previous.isnull()] = 0

#sparse_list = [subskills_sparse, k_traced_sparse, kc_rules_sparse]
#windows = [1,2,3,4,5,6,7,8,9,10]
# create_and_save_sparses(train, sparse_list, windows)
