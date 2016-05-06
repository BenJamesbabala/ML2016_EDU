import numpy as np
import pandas as pd
from scipy.sparse import hstack, coo_matrix, vstack, csr_matrix, lil_matrix
import scipy
import sklearn

from skillsClustering.ClusterSkills import *

def skills_corr_counter_win_v2(ds, sparse_matrix_input, window=None):
    #If window not specified not use window
    student_cfa = ds[['student_id', 'correct_first_attempt']]
    sparse_matrix = csr_matrix(sparse_matrix_input.shapes)

    for col in xrange(sparse_matrix_input.shape[1]):

        skill_indices = np.array(sparse_matrix_input[:,col].nonzero()[0])

        s_cfa = student_cfa.ix[skill_indices]
        grouped = s_cfa.groupby('student_id')

        if window:
            cum = grouped.correct_first_attempt.cumsum()

            cum_df = student_cfa.join(cum, how='left', rsuffix='_cum')

            cum_df = cum_df[['student_id', 'correct_first_attempt_cum']]

            grouped_cum = cum_df.groupby('student_id')

            cum_delay = grouped_cum.shift(window)#.fillna(0)


            diff = cum - cum_delay

            diff_df = pd.merge(student_cfa, diff,
                               right_index=True, left_index=True)

            #diff_fraction =

            diff_fraction = diff_df.shift(1).fillna(0)



            #sg = grouped.apply(cumsum_window, window)
            #sg = sg.reset_index(level=0).drop('student_id',axis=1)
            #if sg.shape[0]==1:
                #sg = sg.transpose()
        else:
            sg = grouped.cumsum()

        sg = sg.sort_index()

        indices = np.array(sg.index)
        values = sg.values

        sparse_matrix[indices,col] = values

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
            if sg.shape[0]==1:
                sg = sg.transpose()
        else:
            sg = grouped.cumsum()
        
        sg = sg.sort_index()

        indices = np.array(sg.index)
        values = sg.values

        sparse_matrix[indices,col] = values

    return sparse_matrix


def cumsum_window(obs, N=5):
    ''' Receives a DF with observations and returns a DF
    of the same size but with a windowed cumsum '''
    cum = obs.cumsum().correct_first_attempt
    cum_delay = cum.shift(N).fillna(0)

    den = np.arange(len(cum))+1
    den[N:] = N

    diff = cum - cum_delay
    diff = diff/den
    diff = diff.shift(1).fillna(0)
    
    return diff

def corrects_incorrects_counter_win(ds,  window=None):
    ''' Receives the dataset and creates a cumulative windowed sum
    for the columns corrects and incorrects '''

    #If window not specified not use window
    student_cfa = ds[['student_id', 'step_id', 'corrects', 'incorrects']]
    grouped = student_cfa.groupby(['student_id', 'step_id'])

    new_df = pd.DataFrame(np.zeros((ds.shape[0], 2)), 
                            columns=['cum_corr', 'cum_incorr'])

    if window:        
        cum = grouped.cumsum()

        cum_df = pd.merge(student_cfa[['student_id', 'step_id']], cum, 
                            right_index=True, left_index=True)
        grouped_cum = cum_df.groupby(['student_id', 'step_id'])

        cum_delay = grouped_cum.shift(window).fillna(0)

        diff = cum - cum_delay

        diff_df = pd.merge(student_cfa[['student_id', 'step_id']], diff, 
                            right_index=True, left_index=True)
        diff_df = diff_df.groupby(['student_id', 'step_id'])
        
        previous_columns = diff_df.shift(1)
        previous_columns = previous_columns.fillna(0)
        previous_columns.columns = ['prev_corr',  'prev_incorr']        

        return (previous_columns.prev_corr, previous_columns.prev_incorr)



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

    small_data_frame = data_frame[['student_id', 'step_id', 'correct_first_attempt']]
    grouped = small_data_frame.groupby(['student_id', 'step_id'])
    shifted = grouped.shift(periods=1)

    map_0_minus1 = {0:-1}
    shifted.replace({'correct_first_attempt':map_0_minus1})
    shifted = shifted.fillna(0)

    return shifted.correct_first_attempt



def create_missing_values_indicators(dataframe, column_name):
    # This function creates indicator features to know if there was a null value for a particular feature and a particular record
    column_copy = pd.DataFrame(dataframe[column_name].isnull())
    new_name = column_name + '_d'
    column_copy.rename(columns={column_name:new_name}, inplace=True)
    dummies_column = column_copy.applymap(lambda x: 1 if x else 0)
    
    return dummies_column


def hints_column(ds, train_indexes):

    ds_filtered = ds[['student_id', 'hints', 'row']]
    train_df = ds_filtered.loc[train_indexes]
    
    train_gr = train_df.groupby('student_id')
    mean_hints = train_gr.hints.mean()
    hints_matrix = mean_hints.reset_index()
    hints_matrix.columns = ['student_id', 'hints_avg']
    
    merged = ds_filtered.merge(hints_matrix, how='left',
                                left_on='student_id', right_on='student_id')

    return merged.hints_avg



  
def sparse_kc_skills(ds, skill_column, opportunity_column):

    #Create temporal columns
    ds.loc[:,'KCop'] = np.array(ds[opportunity_column].str.split('~~'))
    ds.loc[:,'KCop'] = map(list_string_to_int, ds['KCop'])
    ds.loc[:,'KC'] = ds[skill_column].str.split('~~')
    ds.loc[:,'KCzip'] = map(zip,ds.KC,ds.KCop)
    ds.loc[:,'KCdict'] = ds.KCzip.map(dict)

    #Create sparse matrix
    #sparse_ds = pd.DataFrame(list(ds['KCdict']), index = ds.index)
    list_dicts = list(ds['KCdict'])
    v = DictVectorizer(sparse=True)
    sparse_ds = v.fit_transform(list_dicts)
    #Remove temporal columns
    ds.drop('KCop',1, inplace = True)
    ds.drop('KC',1, inplace=True)
    ds.drop('KCzip',1, inplace=True)
    ds.drop('KCdict',1, inplace=True)

    return sparse_ds, v

     


#sparse_list = [subskills_sparse, k_traced_sparse, kc_rules_sparse]
#windows = [1,2,3,4,5,6,7,8,9,10]
# create_and_save_sparses(train, sparse_list, windows)


def main():

    #Define the train/test split
    train_ix, test_ix = splitter(ds)
    #1st define which skills column to be used. 
    #Uncomment the one to be used
    
    skills_mapping = 'kc_subskills'
    #skills_mapping = 'k_traced_skills'
    #skills_mapping = 'k_traced_skills'
    #Define window to use
    window = 5
    #Define if clustering of skills is used:
    clustering = True
    n_clusters = 50


    #Creation of the sparse subskills matrix
    if skills_mapping == 'kc_subskills':
        skills_sparse, skills_vectorizer = sparse_kc_skills(ds,
                                                            'kc_subskills',
                                                            'opp_subskills')
    elif skills_mapping == 'k_traced_skills':
        skills_sparse, skills_vectorizer = sparse_kc_skills(ds,
                                                            'k_traced_skills',
                                                            'opp_k_traced')
    else:
        skills_sparse, skills_vectorizer = sparse_kc_skills(ds,
                                                            'kc_rules',
                                                            'opp_rules')

    #Clustering of skills
    if clustering:
        #
        clusters_dict = clusterDictionary(ds, skills_mapping, 
                                    number_clusters=n_clusters)

        #Shrink the sparse matrix using the cluster of skills
        skills_sparse_cl = sparse_matrix_clusterer(skills_sparse,
                                                    skills_vectorizer,
                                                    clusters_dict)    
    
    #Apply a cumulative window to the skills sparse matrix        
    cumulative_skills_sparse = skills_corr_counter_win(ds, skills_sparse_cl, window=window)
    
    #Create previous CFA column
    prev_cfa = previous_correct_first_attempt_column(ds)   
    ds['prev_cfa'] = prev_cfa
    
    #Create cumulative columns for corrects and incorrects columns
    prev_corr, prev_incorr= corrects_incorrects_counter_win(ds, window=window)
    ds['prev_corr'] = prev_corr
    ds['prev_incorr'] = prev_incorr

    #Create hints column
    hints_rate = hints_column(ds, train_ix)
    ds['hints_rate'] = hints_rate




if __name__ == '__main__':
    main()





