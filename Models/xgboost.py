import xgboost as xgb
import numpy as np

#dtrain = xgb.DMatrix( X_train, y_train)
#dval = xgb.DMatrix(X_val, y_val)
#
#dxtest = xgb.DMatrix( X_test)
#
#param = { 'booster':'gbtree','eval_metric':'logloss', 'silent':0, 
#        'tree_method':'exact','bst:alpha':1e-2, 'bst:eta':0.1, 
#        'bst:scale_pos_weight':0.171158, 'max_depth':1, 
#        'nthread':20}
#
#evallist  = [(dval,'eval'), (dtrain,'train')]
#num_rounds = 10
#bst = xgb.train( param, dtrain, num_rounds, evallist )
#
#
#
#dtrain = xgb.DMatrix( X_train, y_train)
#dval = xgb.DMatrix(X_val, y_val)
#
#train_rmse, train_ll, val_rmse, val_ll = gridsearch_xgboost(dtrain, dval, y_train, y_val)




#bst = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, 
#                n_estimators=1000, silent=False, 
#                objective='binary:logistic', nthread=-1, 
#                gamma=0, min_child_weight=1, max_delta_step=0, 
#                subsample=1, colsample_bytree=1, colsample_bylevel=1, 
#                reg_alpha=10e-2, reg_lambda=1, scale_pos_weight=1, 
#                base_score=0.171158, seed=0, missing=None)
#
#bst.fit(X_train, y_train, eval_metric='logloss')

#param = { 'booster':'gbtree','eval_metric':'logloss', 
#        'tree_method':'exact' }

#param['eval_metric'] = ['logloss']
#param['tree_method'] = 'exact'

#evallist  = [(dval,'eval'), (dtrain,'train')]

#bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=100, 
#            evals=evallis)

#bst = xgb.train( param, dtrain, num_round, evallist )





pred_proba_train = bst.predict(X_train)

mse_train = mean_squared_error(y_train, pred_proba_train)
rmse_train = np.sqrt(mse_train)
logloss_train = log_loss(y_train, pred_proba_train)

#Evaluation in validation set
pred_proba_val = bst.predict(X_val)

mse_val = mean_squared_error(y_val, pred_proba_val)
rmse_val = np.sqrt(mse_val)
logloss_val = log_loss(y_val, pred_proba_val)

rmse_train
rmse_val
logloss_train
logloss_val


def gridsearch_xgboost(dtrain, dval, y_train, y_val):
    
    depths = [3,20,50,100]
    alphas = np.logspace(-7,0,7)

    train_rmse = {}
    train_ll = {}
    val_rmse = {}
    val_ll = {}

    for depth in depths:
        for alpha in alphas:

            param = { 'booster':'gbtree','eval_metric':'logloss', 'silent':0, 
                    'tree_method':'exact','bst:alpha':alpha, 'bst:eta':0.1, 
                    'bst:scale_pos_weight':0.171158, 'max_depth':depth, 
                    'nthread':20}
            
            evallist  = [(dval,'eval'), (dtrain,'train')]
            num_rounds = 10
            bst = xgb.train( param, dtrain, num_rounds, evallist )
        
            pred_proba_train = bst.predict(dtrain)
        
            mse_train = mean_squared_error(y_train, pred_proba_train)
            rmse_train = np.sqrt(mse_train)
            logloss_train = log_loss(y_train, pred_proba_train)
            
            train_rmse[(depth, alpha)] = rmse_train
            train_ll[(depth, alpha)] = logloss_train
            #Evaluation in validation set
            pred_proba_val = bst.predict(dval)
            
            mse_val = mean_squared_error(y_val, pred_proba_val)
            rmse_val = np.sqrt(mse_val)
            logloss_val = log_loss(y_val, pred_proba_val)
        
            val_rmse[(depth, alpha)] = rmse_val
            val_ll[(depth, alpha)] = logloss_val

    return train_rmse, train_ll, val_rmse, val_ll


"""
---------------------------------------------------
import numpy as np
X_train = np.load('X_train.npy').item()
X_test = np.load('X_test.npy').item()
y_test = np.load('y_test.npy')
y_train = np.load('y_train.npy')

----------------------------------------------
#Evaluation

First Model

[299]   eval-logloss:0.369825   train-logloss:0.356933
[299]   eval-rmse:0.336063  train-rmse:0.331482


Second Model (Super X Small)

[299]   eval-rmse:0.318282  train-rmse:0.317684


Third Model (Super X Small 6000 iterations)

[1930]  eval-rmse:0.312692  train-rmse:0.304757

-------------------------------------------------------
#Plotting 

fraction_of_positives, mean_predicted_value = calibration_curve(y_test2, ypred, normalize=False, n_bins=50)
plt.plot([0,1],[0,1], 'k:', label='Perfectly Calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Actual")
plt.ylabel("Fraction of positives")
ax1.legend(loc="lower right")
plt.title('Calibration plots  (reliability curve)')


"""


