run __init__.py
import xgboost as xgb

ds = load_ds('./Datasets/algebra_2008_2009/ds_featurized_f.txt')
train_ix, test_ix = splitter(ds)
train_ix, val_ix = splitter(ds.ix[train_ix])

latent = pd.read_csv('./Datasets/algebra_2008_2009/latent_df_gs', 
                        index_col=0, sep='\t')



saved = np.load('./sparse_matrices/cluster_75_window_10/cum_skills_sparse_1.npy')
cum_skills_sparse1 = saved.item()
saved = np.load('./sparse_matrices/cluster_75_window_10/cum_skills_sparse_2.npy')
cum_skills_sparse2 = saved.item()
saved = np.load('./sparse_matrices/cluster_75_window_10/cum_skills_sparse_3.npy')
cum_skills_sparse3 = saved.item()
del saved

# RECOVER BETA AND THETA PARAMETERS
saved = np.load('./Models/baseline_beta_prob.npy')
beta_prob = saved[0]
step_idx = saved[1]

saved = np.load('./Models/baseline_theta_stud.npy')
theta_stud = saved[0]
stud_idx = saved[1]

 #create theta and beta DFs
beta = pd.DataFrame(np.array([beta_prob, step_idx]).T, 
                    columns=['beta','step_id'])
theta = pd.DataFrame(np.array([theta_stud, stud_idx]).T, 
                    columns=['theta','student_id'])

ds = merge_estimates_w_data(ds,beta, theta)


#REMOVE NON USEFUL COLUMNS
non_num_features = get_non_numeric_features(ds)
ds_num = ds.drop(non_num_features, 1)
X_ds = ds_num.drop(['correct_first_attempt', 'y_one_negative_one',
                    'incorrects', 'hints', 'corrects'], 1)
not_used_columns = [u'step_duration', u'correct_step_duration', u'error_step_duration',
                     u'row']
X_ds = X_ds.drop(not_used_columns, 1)
#Concat sparse matrix and dataframe
X = concat_sparse_w_df(cum_skills_sparse1, X_ds)
X = csr_matrix(hstack([X, cum_skills_sparse2]))
X = csr_matrix(hstack([X, cum_skills_sparse3]))

#Concat with latent
latent_matrix = latent.as_matrix()
X = csr_matrix(hstack([X, latent_matrix]))


#Split X in train and validation
X_train = X[train_ix]
X_val = X[val_ix]
y = ds_num.correct_first_attempt
y_train = y.loc[train_ix]
y_val = y.loc[val_ix]
y_test = y.loc[test_ix]
X_test = X[test_ix]

#xgboost
dtrain = xgb.DMatrix( X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)


