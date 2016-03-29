import pandas as pd 
import numpy as np 


def get_nulls_by_column(ds, filter='only_null'):
    #Return columns and its null count
    if type(ds) != pd.DataFrame:
        raise TypeError('ds must be pd.DataFrame')

    for col in ds.columns:
        null_sum = ds[col].isnull().sum()
        if null_sum or filter != 'only_null':
            print col, ds[col].isnull().sum()


def valid_error_step_duration(ds):
    #Return the index of elements
    #which contain a valid NaN the step was solved
    #correctly by the student.
    if type(ds) != pd.DataFrame:
        raise TypeError('ds must be pd.DataFrame')

    columns = ['Error Step Duration (sec)','Correct First Attempt']
    ds_sub = ds[columns]

    ESD_null = ds_sub[ds_sub['Error Step Duration (sec)'].isnull()]
    ESD_missing_null = ESD_null[ESD_null['Correct First Attempt']==1]

    return ESD_missing_null.index

def set_value_for_index_column(ds, index, column, value):
    if type(ds) != pd.DataFrame:
        raise TypeError('ds must be pd.DataFrame')
    ds[column].ix[index] = value
    
def fill_KC_null(ds):


def main():
    
    train = pd.read_csv('./Datasets/algebra_2008_2009/algebra_2008_2009_train.txt', sep='\t')
    test = pd.read_csv('./Datasets/algebra_2008_2009/algebra_2008_2009_test.txt', sep='\t')

    #Dataset contains a column called Error Step Duration which can be NaN
    #if there is a missing value or if the step was solved correctly (valid NaN). 
    #Set the value of valid NaNs to -1
    set_value_for_index_column(ds, valid_error_step_duration(ds), 'Error Step Duration (sec)',-1)

    

if __name__ == '__main__':
    main()




