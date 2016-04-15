import pandas as pd

def splitter(data_frame):
    #data_frame: pandas dataframe

    test_indexes = list() 
    ds_grouped = data_frame.groupby(['Anon Student Id', 'Unit'])
    indices = ds_grouped.indices

    for student, unit in indices:

            indexes_student_unit = indices[student,unit]

            if len(indexes_student_unit)<2:
                pass

            else:

                data_unit_student = data_frame.ix[indices[student,unit]]
                problem_name = data_unit_student['Problem Name'][data_unit_student.index[-1]]
                data_last_problem = data_unit_student[data_unit_student['Problem Name']==problem_name]
                last_view = data_last_problem['Problem View'][data_last_problem.index[-1]]
                data_last_view = data_last_problem[data_last_problem['Problem View']==last_view]
                test_indexes.extend(data_last_view.index)

    return test_indexes
