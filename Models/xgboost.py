import xgboost as xgb

dtrain = xgb.DMatrix( X_train, y_train)
dtest = xgb.DMatrix( X_test, y_test)

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = ['logloss', 'rmse']
param['tree_method'] = 'exact'

evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 6000
bst = xgb.train( param, dtrain, num_round, evallist )


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

[299]	eval-logloss:0.369825	train-logloss:0.356933
[299]	eval-rmse:0.336063	train-rmse:0.331482


Second Model (Super X Small)

[299]	eval-rmse:0.318282	train-rmse:0.317684


Third Model (Super X Small 6000 iterations)

[1930]	eval-rmse:0.312692	train-rmse:0.304757

-------------------------------------------------------
#Plotting 

fraction_of_positives, mean_predicted_value = calibration_curve(y_test2, ypred, normalize=False, n_bins=50)
plt.plot([0,1],[0,1], 'k:', label='Perfectly Calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Actual")
plt.ylabel("Fraction of positives")
ax1.legend(loc="lower right")
plt.title('Calibration plots  (reliability curve)')
"""
