import pandas as pd

def splitter(data_frame):
    #data_frame: pandas dataframe

    test_indices = list() 
    ds_grouped = data_frame.groupby(['student_id', 'unit'])
    indices = ds_grouped.indices

    for student, unit in indices:

            indices_student_unit = indices[student,unit]
            data_unit_student = data_frame.ix[indices_student_unit]

            if len(data_unit_student.problem_name.unique())<2:
                pass

            else:

                data_unit_student = data_frame.ix[indices_student_unit]
                problem_name = data_unit_student['problem_id'][data_unit_student.index[-1]]
                data_last_problem = data_unit_student[data_unit_student['problem_id']==problem_name]
                last_view = data_last_problem['view'][data_last_problem.index[-1]]
                data_last_view = data_last_problem[data_last_problem['view']==last_view]
                test_indices.extend(data_last_view.index)

    all_indices = list(data_frame.index)
    train_indices = list(set(all_indices)-set(test_indices))


    return train_indices, test_indices



