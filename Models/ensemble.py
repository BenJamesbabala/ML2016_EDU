ensemble_ds = pd.DataFrame(np.array([base_pred, latent_pred, y_val]).T, 
                            columns=['base_pred', 'latent_pred', 'y'])



lr_ensemble = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                                fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, 
                                solver='liblinear', max_iter=100, 
                                multi_class='ovr', verbose=0, 
                                warm_start=False, n_jobs=4)

lr_ensemble.fit(ensemble_ds.drop('y',1), ensemble_ds.y)
e_pred_proba = lr_ensemble.predict_proba(ensemble_ds.drop('y',1))

e_pred_proba_1 = [x[1] for x in e_pred_proba]


mse = mean_squared_error(y_val, e_pred_proba_1)
logloss = log_loss(y_val, e_pred_proba_1)


mse_train = mean_squared_error(y_train, pred_proba_train_1)
rmse_train = np.sqrt(mse_train)
logloss_train = log_loss(y_train, pred_proba_train_1)

#Evaluation in validation set
pred_proba_val = lr.predict_proba(X_val)
pred_proba_val_1 = [x[1] for x in pred_proba_val]

mse_val = mean_squared_error(y_val, pred_proba_val_1)
rmse_val = np.sqrt(mse_val)
logloss_val = log_loss(y_val, pred_proba_val_1)