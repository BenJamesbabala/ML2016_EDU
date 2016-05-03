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

    small_data_frame = train[['student_id', 'step_id', 'correct_first_attempt']]
    grouped = small_data_frame.groupby(['student_id', 'step_id'])
    shifted = grouped.shift(periods=1)
    data_frame['cfa_previous'] = shifted.correct_first_attempt



def create_missing_values_indicators(dataframe, column_name):
    # This function creates indicator features to know if there was a null value for a particular feature and a particular record
    column_copy = pd.DataFrame(dataframe[column_name].isnull())
    new_name = column_name + '_d'
    column_copy.rename(columns={column_name:new_name}, inplace=True)
    dummies_column = column_copy.applymap(lambda x: 1 if x else 0)
    
    return dummies_column


#sparse_list = [subskills_sparse, k_traced_sparse, kc_rules_sparse]
#windows = [1,2,3,4,5,6,7,8,9,10]
# create_and_save_sparses(train, sparse_list, windows)


def main():

    #1st define which skills column to be used
    subskills_sparse, subskills_vectorizer = sparse_kc_skills(train, 'kc_subskills','opp_subskills')
    k_traced_sparse, k_traced_vectorizer = sparse_kc_skills(train, 'k_traced_skills','opp_k_traced')
    kc_rules_sparse, kc_rules_vectorizer = sparse_kc_skills(train, 'kc_rules','opp_rules')

    



if __name__ == '__main__':
    main()


