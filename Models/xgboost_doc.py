
#xgb.XGBRegressor(self, max_depth=3, learning_rate=0.1, 
#                n_estimators=100, silent=False, 
#                objective='binary:logistic', nthread=-1, 
#                gamma=0, min_child_weight=1, max_delta_step=0, 
#                subsample=1, colsample_bytree=1, colsample_bylevel=1, 
#                reg_alpha=10e-2, reg_lambda=1, scale_pos_weight=1, 
#                base_score=0.171158, seed=0, missing=None)
#
#param = { 'booster':'gbtree','eval_metric':'logloss', 
#        'tree_method':'exact' }
#
#
#
#
#gb.train(params=param, dtrain=dtrain, num_boost_round=100, 
#            evals=evallis)
#
#
#
#
#param = { 'booster':'gbtree','eval_metric':'logloss', 
#        'tree_method':'exact','bst:alpha':1e-2, 'bst:eta':0.1, 
#        'bst:scale_pos_weight':0.171158}
#
#evallist  = [(dval,'eval'), (dtrain,'train')]
#num_rounds = 1000
#bst = xgb.train( param, dtrain, num_rounds, evallist )
##param['eval_metric'] = ['logloss']
##param['tree_method'] = 'exact'
#
##evallist  = [(dval,'eval'), (dtrain,'train')]
#
##bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=100, 
##            evals=evallis)
#
##bst = xgb.train( param, dtrain, num_round, evallist )
#
#
##########################
##   XGBoost Parameters
##########################
#
#xgb.XGBRegressor(self, max_depth=3, learning_rate=0.1, 
#                n_estimators=100, silent=True, 
#                objective='reg:linear', nthread=-1, 
#                gamma=0, min_child_weight=1, max_delta_step=0, 
#                subsample=1, colsample_bytree=1, colsample_bylevel=1, 
#                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
#                base_score=0.5, seed=0, missing=None)
#
#bst.fit(X, y, eval_set=None, eval_metric=None,
#        early_stopping_rounds=None, verbose=True)
#
#
#
#xgboost.train(params, dtrain, num_boost_round=10, evals=(), 
#                obj=None, feval=None, maximize=False, 
#                early_stopping_rounds=None, evals_result=None, 
#                verbose_eval=True, learning_rates=None,
#                xgb_model=None)
#
#*************************
#General Parameters
#*************************
#booster [default=gbtree]
#    gbtree
#    gblinear
#
#*************************
#Booster Parameters
#*************************
#
#eta [default=0.3]
#    Typical final values to be used: 0.01-0.2
#    step size shrinkage used in update to prevents 
#    overfitting.
#
#min_child_weight [default=1]
#    Defines the minimum sum of weights of all 
#    observations required in a child.
#
#max_depth [default=6]
#    Prevent overfitting
#
#max_leaf_nodes
#    The maximum number of terminal nodes or
#     leaves in a tree.
#     Can be defined in place of max_depth. 
#     Since binary trees are created, a 
#     depth of ‘n' would produce a maximum of 2^n leaves.
#
#gamma [default=0]
#    A node is split only when the resulting split
#    gives a positive reduction in the loss function. 
#    Gamma specifies the minimum loss reduction required 
#    to make a split.
#
#max_delta_step [default=0]
#subsample [default=1]
#    Denotes the fraction of observations 
#    to be randomly samples for each tree.
#    Typical values: 0.5-1
#
#colsample_bytree [default=1]
#    subsample ratio of columns for each split, 
#    in each level.
#
#
#colsample_bylevel [default=1]
#    Denotes the subsample ratio of columns for each split, 
#    in each level.
#
#lambda [default=1]
#    L2 regularization term on weights 
#    (analogous to Ridge regression)
#
#alpha [default=0]
#    L1 regularization term on weight 
#    (analogous to Lasso regression)
#
#scale_pos_weight [default=1]
#    A value greater than 0 should be used in case 
#    of high class imbalance as it helps in faster 
#    convergence.
#     A typical value to consider: 
#     sum(negative cases) / sum(positive cases)
#
#
#tree_method, string [default='auto']
#    ‘auto': Use heuristic to choose faster one.
#    ‘exact': Exact greedy algorithm.
#    ‘approx': Approximate greedy algorithm using 
#    sketching and histogram.
#
#sketch_eps, [default=0.03]
#    
# 
#
#*************************
#Learning Task Parameters
#*************************
#
#objective [default=reg:linear]
#    'reg:linear' -linear regression
#    'reg:logistic' -logistic regression
#    'binary:logistic' -logistic regression for binary 
#        classification, output probability
#    'binary:logitraw' -logistic regression for binary 
#        classification, output score before logistic 
#        transformation
#    'count:poisson' -poisson regression for count data, 
#        output mean of poisson distribution
#        max_delta_step is set to 0.7 by default in poisson 
#        regression (used to safeguard optimization)
#    'multi:softmax' -set XGBoost to do multiclass 
#        classification using the softmax objective, 
#        you also need to set num_class(number of classes)
#    'multi:softprob' -same as softmax, but output a     
#        vector of ndata * nclass, which can be further reshaped
#        to ndata, nclass matrix. The result contains predicted
#        probability of each data point belonging to each class.
#    'rank:pairwise' -set XGBoost to do ranking task by 
#        minimizing the pairwise loss
#
#eval_metric [ default according to objective ]
#    rmse - root mean square error
#    mae - mean absolute error
#    logloss - negative log-likelihood
#    error - Binary classification error rate (0.5 threshold)
#    merror - Multiclass classification error rate
#    mlogloss - Multiclass logloss
#    auc: Area under the curve
#
#base_score [ default=0.5 ]
#
#
#eta -> learning_rate
#lambda -> reg_lambda
#alpha -> reg_alpha
#