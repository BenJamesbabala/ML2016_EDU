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
    #Set the same value for all the items in ds.loc[(column,index)] inplace

    if type(ds) != pd.DataFrame:
        raise TypeError('ds must be pd.DataFrame')
    ds[column].ix[index] = value
    

def fill_KC_null(ds, column):
    #Fill null values in KC column by the string 'null_unit'
    #taking unit value from ds['Problem Hierarchy']

    ds_na_KC = ds[ds[column].isnull()]

    units = ds_na_KC['Problem Hierarchy']
    units_str = units.astype(str)

    fill_KC = pd.Series(['null_'+s for s in units_str.values],index=units_str.index)

    ds.loc[(ds_na_KC.index,column)] = fill_KC

    return ds


def unit_to_int(ds,test = False):
    #Replace unit strings to integers in a one-to-one mapping

    units_str = ds['Problem Hierarchy'].unique()
    units_int = range(len(units_str))
    mapping_dict = dict(zip(units_str,units_int))

    if test:       
       return ds.replace({'Problem Hierarchy':mapping_dict}) , test.replace({'Problem Hierarchy':mapping_dict}) 

    return ds.replace({'Problem Hierarchy':mapping_dict})  


def split_problem_hierarchy(ds):

    if type(ds) != pd.DataFrame:
        raise TypeError('ds must be pd.DataFrame')

    hierarchy = ds['Problem Hierarchy']
    ds.drop('Problem Hierarchy',1,inplace=True)

    hierarchy = hierarchy.apply(lambda x: str.split(x,',') )
    unit = pd.Series([u[0] for u in hierarchy.values],index=hierarchy.index)
    section = pd.Series([s[1] for s in hierarchy.values],index=hierarchy.index)

    ds['Unit'] = unit
    ds['Section'] = section

    return ds

def create_unique_step_id(ds):
    grouped = train.groupby(['Unit','Section','Problem Name', 'Step Name'])
    groups = grouped.groups
    #groups is a dictionary

def main():
    
    train = pd.read_csv('./Datasets/algebra_2008_2009/algebra_2008_2009_train.txt', sep='\t')
    test = pd.read_csv('./Datasets/algebra_2008_2009/algebra_2008_2009_test.txt', sep='\t')

    #Dataset contains a column called Error Step Duration which can be NaN
    #if there is a missing value or if the step was solved correctly (valid NaN). 
    #Set the value of valid NaNs to -1
    set_value_for_index_column(train, valid_error_step_duration(train), 'Error Step Duration (sec)',-1)
    train = split_problem_hierarchy(train)

    train = unit_to_int(train)
    train = fill_KC_null(train, 'KC(SubSkills)')



if __name__ == '__main__':
    main()




