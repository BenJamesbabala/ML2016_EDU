import numpy as np
import pandas as pd






def remove_unused_columns(ds):

    unused_columns = [u'problem_name', u'view', u'step_name',
       u'start_time', u'first_trans_time', u'correct_trans_time', u'end_time',
       u'step_duration', u'correct_step_duration', u'error_step_duration',
       u'correct_first_attempt', u'incorrects', u'hints', u'corrects',
       u'kc_subskills', u'opp_subskills', u'k_traced_skills', u'opp_k_traced',
       u'kc_rules', u'opp_rules', u'unit', u'section', u'problem_id']

    return ds.drop(unused_columns,1)


def create_dummy_representation(ds):
    cols = [u'student_id', u'step_id']
    #cols = [u'student_id']
    return pd.get_dummies(ds, columns = cols, sparse=True)




def main():

    train_lr = remove_unused_columns(train)
    train_lr_d = create_dummy_representation(train_lr)

    sub_students = np.random.choice(students, 200)


def encoding_typestr(a):
    return str(type(a)) == "<type 'str'>"



if __name__ == '__main__':
    main()
