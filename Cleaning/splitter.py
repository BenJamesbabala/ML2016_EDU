import pandas as pd


def splitter(data):
    #data: pandas dataframe

    students = pd.unique(data["Anon Student Id"].ravel())
    units = pd.unique(data["Unit"].ravel())
    row_numbers = list()

    for student in students:
        for unit in units:
            i = -1
            current_indexes = list()

            if data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)].empty:
                pass
            elif pd.unique(data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)]["Problem Name"]).shape[0] == 1:
                pass
            else:
                problem_name = data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)][i:]["Problem Name"].ravel()[0]

                while 1:
                    if data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)][i:]["Problem Name"].ravel()[0] == problem_name:
                        current_indexes.append(data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)][i:].index[0])
                        i += - 1
                    else:
                        break

                row_numbers.append((min(current_indexes), max(current_indexes)))

    return row_numbers



def splitter1(data):
    #data: pandas dataframe

    students = pd.unique(data["Anon Student Id"].ravel())
    units = pd.unique(data["Unit"].ravel())
    test_indexes = list()

    for student in students:
        for unit in units:
            
            current_indexes = list()

            data_unit_student = data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)]


            if data_unit_student.empty or (data_unit_student.shape[0]==1):
                pass
            
            else:
                problem_name = data_unit_student['Problem Name'][data_unit_student.index[-1]]

                data_last_problem = data_unit_student[data_unit_student['Problem Name']==problem_name]
                last_view = data_last_problem['Problem View'][data_last_problem.index[-1]]

                data_last_view = data_last_problem[data_last_problem['Problem View']==last_view]

                test_indexes.extend(data_last_view.index)

    return test_indexes


def splitter2(data):
    #data: pandas dataframe


    test_indexes = list() 

    ds_grouped = train.groupby(['Anon Student Id', 'Unit'])
    

    indices = ds_grouped.indices
    

    for student, unit in indices:
            

            indexes_student_unit = indices[student,unit]

            if len(indexes_student_unit)<2:
                pass
            else:

                data_unit_student = data.ix[indices[student,unit]]
                
                problem_name = data_unit_student['Problem Name'][data_unit_student.index[-1]]

                data_last_problem = data_unit_student[data_unit_student['Problem Name']==problem_name]
                last_view = data_last_problem['Problem View'][data_last_problem.index[-1]]

                data_last_view = data_last_problem[data_last_problem['Problem View']==last_view]

                test_indexes.extend(data_last_view.index)

    return test_indexes