import pandas as pd


def splitter(data):

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

                while 1 != 0:
                    if data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)][i:]["Problem Name"].ravel()[0] == problem_name:
                        current_indexes.append(data[(data["Anon Student Id"] == student) & (data["Unit"] == unit)][i:].index[0])
                        i += - 1
                    else:
                        break

                row_numbers.append((min(current_indexes), max(current_indexes)))

    return row_numbers
