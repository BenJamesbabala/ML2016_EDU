

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



def skills_corr_counter(ds, features, sparse_matrix_input, subskills_col):

    #This was the second version. The part that is still too slow is the filtering and groupby of the ds
    students = ds.student_id.unique()
    students_group = ds.groupby('student_id')

    sparse_matrix = coo_matrix([])

    for col in range(sparse_matrix_input.shape[1]):

        new_column = pd.DataFrame(np.zeros(ds.shape[0]))

        skill_indexes = np.array(sparse_matrix_input[:,col].nonzero()[0])

        for student in students:
            student_index = np.array(students_group.groups[student])
            indexes = np.intersect1d(skill_indexes, student_index, assume_unique=True)
            new_column.loc[indexes] = ds.ix[indexes].groupby('student_id').cumsum().correct_first_attempt.reshape((indexes.shape[0], 1))

        sparse_col = coo_matrix(new_column)


        if sparse_matrix.size>0:
            sparse_matrix = hstack([sparse_matrix, sparse_col])
        else:
            sparse_matrix = sparse_col
        print 'done column'