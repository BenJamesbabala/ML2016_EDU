from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor



def main():



    ab = AdaBoostRegressor(base_estimator=None, n_estimators=50, 
                            learning_rate=1.0, loss='exponential', 
                            random_state=None)  

    ab.fit(X_train, y_train)

    #Evaluation in train set
    #Evaluation in train set
    pred_proba_train = ab.predict(X_train)
        
    mse_train = mean_squared_error(y_train, pred_proba_train)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y_train, pred_proba_train)
    
    #Evaluation in validation set
    pred_proba_val = ab.predict(X_val)
        
    mse_val = mean_squared_error(y_val, pred_proba_val)
    rmse_val = np.sqrt(mse_val)
    logloss_val = log_loss(y_val, pred_proba_val)
    
    rmse_train
    rmse_val
    logloss_train
    logloss_val




if __name__ == '__main__':
    main()