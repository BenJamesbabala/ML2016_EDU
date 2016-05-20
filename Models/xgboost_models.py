import xgboost as xgb
import numpy as np

#dtrain = xgb.DMatrix( X_train, y_train)
#dval = xgb.DMatrix(X_val, y_val)
#
#dxtest = xgb.DMatrix( X_test)
##'scale_pos_weight':0.171158,
##'eval_metric':'logloss'
#param = { 'booster':'gbtree', 'silent':0, 
#        'tree_method':'exact','lambda':10e-3, 'eta':0.1, 
#         'max_depth':50, 'objective':'binary:logitraw'}
#
#evallist  = [(dval,'eval'), (dtrain,'train')]
#num_rounds = 100
#bst = xgb.train( param, dtrain, num_rounds, evallist )
#
#
#
#dtrain = xgb.DMatrix( X_train, y_train)
#dval = xgb.DMatrix(X_val, y_val)
#
#train_rmse, train_ll, val_rmse, val_ll = gridsearch_xgboost(dtrain, dval, y_train, y_val)
#
#
#param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logitraw' }
#param['nthread'] = 20
#param['eval_metric'] = ['logloss']
#param['tree_method'] = 'exact'
#
#evallist  = [(dval,'eval'), (dtrain,'train')]
#
#num_round = 200


#bst = xgb.train( param, dtrain, num_round, evallist )


########################
# XGBOOST GRIDSEARCH
########################

def gridsearch_xgboost(dtrain, dval, y_train, y_val):
    
    depths = [3,10,50]
    
    depth=2
    min_child_weight = 5
    alphas = np.logspace(-7,1,4)
    colsample = 0.5

    train_rmse = {}
    train_ll = {}
    val_rmse = {}
    val_ll = {}

    for depth in depths:
        for alpha in alphas:

            param = { 'booster':'gbtree','eval_metric':'logloss', 'silent':0, 
                    'tree_method':'exact','lambda':alpha, 'eta':0.1, 
                    'max_depth':depth, 'objective':'binary:logistic', 
                    'min_child_weight':min_child_weight, 
                    'colsample_bytree':colsample }
                    #'n_estimators':10000}
                    #
                    #'nthread':1}
            
            evallist  = [(dval,'eval'), (dtrain,'train')]
            num_rounds = 1000
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




def main():

    ########################
    # XGBOOST NORMAL FIT
    ########################

    depth=20
    min_child_weight = 1
    colsample = 1
    alpha = 9.9999999999999995e-08

    param = { 'booster':'gbtree','eval_metric':'logloss', 'silent':0, 
            'tree_method':'exact','lambda':alpha, 'eta':0.1, 
            'max_depth':depth, 'objective':'binary:logistic'}
            #'n_estimators':10000}
            #
            #'nthread':1}
    
    evallist  = [(dval,'eval'), (dtrain,'train')]
    num_rounds = 750
    bst = xgb.train( param, dtrain, num_rounds, evallist )
    
    pred_proba_train = bst.predict(dtrain)
    
    mse_train = mean_squared_error(y_train, pred_proba_train)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y_train, pred_proba_train)
    
    #Evaluation in validation set
    pred_proba_val = bst.predict(dval)
    
    mse_val = mean_squared_error(y_val, pred_proba_val)
    rmse_val = np.sqrt(mse_val)
    logloss_val = log_loss(y_val, pred_proba_val)



if __name__ == '__main__':
    main()


