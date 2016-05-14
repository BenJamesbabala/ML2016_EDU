import pandas as pd 
import numpy as np 
import graphlab as gl


def getLatents(trainingData, user = 'student_id', 
                item = 'step_id',targ = 'correct_first_attempt',
                allVariables = False,user_data = None, 
                item_data=None ,num_factors = 8, 
                max_iterations =100):
    '''
    input: pandas dataframe, need columns names for user/item uses default student_id and step_id
        CAN provide pandas df that summarizes users and or items
    output: two pandas dataframe and the model that produces them- one df for users one df for items, holds latent variables and linear term
    model has .predict() function that you can feed a df with same format into, will return df of scores and ranking for problems/students
    '''
    itemDF = pd.DataFrame()
    userDF = pd.DataFrame()
    trainingData = trainingData[['step_id', 'student_id', 'correct_first_attempt']]


    if user_data !=None:
        user_data = gl.SFrame(user_data)
    if item_data !=None:
        user_data = gl.SFrame(item_data)

    if allVariables==False:
        #dont use extra data
        trainingData = trainingData[[item,user,targ]]

    sf = gl.SFrame(trainingData)
    model =gl.recommender.factorization_recommender.create( sf , user_id= user,item_id = item, 
                                                            target = targ, num_factors =num_factors,
                                                            user_data=user_data, item_data=item_data, 
                                                            max_iterations =max_iterations,
                                                            verbose =True, binary_target=True, 
                                                            linear_regularization=1e-7,
                                                            regularization = 1e-6)
    
    #recomendations= model.recommend()

    latentVariablesItem = model['coefficients'][item]['factors']
    latentVariablesUser = model['coefficients'][user]['factors']

    #separate latent features into separate columns
    for i in range(len(latentVariablesItem[0])):
        itemDF['itemLatent'+ str(i)]=latentVariablesItem.apply(lambda x: x[i])
        userDF['userLatent'+ str(i)]=latentVariablesUser.apply(lambda x: x[i])

    
    itemDF['step_id']= pd.Series(model['coefficients'][item]['step_id'])
    userDF['student_id']= pd.Series(model['coefficients'][user]['student_id'])


    return itemDF, userDF , model


def factorsToMergeWithData(data,itemFactors, userFactors):
    data= data[['student_id','step_id']]

    merged = data.merge( itemFactors, how = 'left',left_on='step_id', right_on='step_id')
    merged = merged.merge(userFactors,how = 'left',left_on='student_id',right_on='student_id')
        
    for i in range(itemFactors.shape[1]-1):
        merged['cross'+str(i)] = merged[str('itemLatent'+ str(i))] * merged[str('userLatent'+ str(i))]  
        
    merged = merged.drop(['student_id','step_id'],axis=1)
    merged = merged.fillna(0)
    
    return merged


def main():
    #location = '/home/michael/Desktop/machineLearningProject/27042016_train.txt' 
    #data = pd.read_csv(location,sep ='\t',nrows =10 ,skipinitialspace =False)
    
    itemDF,userDF, model = getLatents(ds.ix[train_ix], user = 'student_id', 
                                        item = 'step_id',targ = 'correct_first_attempt',
                                        allVariables = False, num_factors = 8)

    #print model['coefficients']
    #print( itemDF.shape )
    
    latent_df = factorsToMergeWithData(ds, itemDF, userDF)
    # latent_df.to_csv('./Datasets/algebra_2008_2009/latent_df_gs', sep='\t')




if __name__ == '__main__':
        main()




#X_val = ds.ix[val_ix][['step_id','student_id']]
#X_val_sf = gl.SFrame(X_val)
#pred_sf = model.predict(X_val_sf)
#pred = np.array(pred_sf)
#
#
#
#y_val = ds.ix[val_ix].correct_first_attempt

#EVALUATION


def latent_x_validation(train, val):

    #train_rmse, train_ll, val_rmse, val_ll = latent_x_validation(ds.ix[train_ix], 
    #                                                             ds.ix[val_ix])

    itemDF = pd.DataFrame()
    userDF = pd.DataFrame()
    trainingData = train[['step_id', 'student_id', 'correct_first_attempt']]
    valData = val[['step_id', 'student_id', 'correct_first_attempt']]

    sf = gl.SFrame(trainingData)
    val_sf = gl.SFrame(valData)

    y_train = trainingData.correct_first_attempt
    y_val = valData.correct_first_attempt

    user = 'student_id'
    item = 'step_id'
    targ = 'correct_first_attempt'
    allVariables = False
    user_data = None
    item_data=None
    max_iterations =100

    num_factors = [8,50,100]
    linear_regularization = np.logspace(-7,-1,7)
    regularization= np.logspace(-7,-1,7)

    train_rmse = []
    train_ll = []
    val_rmse = []
    val_ll = []

    for factor in num_factors:
        for lin_reg in linear_regularization:
            for reg in regularization:
                model =gl.recommender.factorization_recommender.create( sf , user_id=user, item_id=item, 
                                                            target = targ, num_factors=factor,
                                                            user_data=user_data, item_data=item_data, 
                                                            max_iterations=max_iterations,
                                                            verbose =True, binary_target=True, 
                                                            linear_regularization=lin_reg,
                                                            regularization = reg)

                #Evaluation in train set
                pred_proba_train = model.predict(sf)
                pred_proba_train = np.array(pred_proba_train)
                
                mse_train = mean_squared_error(y_train, pred_proba_train)
                rmse_train = np.sqrt(mse_train)
                logloss_train = log_loss(y_train, pred_proba_train)

                train_rmse.append(rmse_train)
                train_ll.append(logloss_train)
                
                #Evaluation in validation set
                pred_proba_val = model.predict(val_sf)
                pred_proba_val = np.array(pred_proba_val)
                
                mse_val = mean_squared_error(y_val, pred_proba_val)
                rmse_val = np.sqrt(mse_val)
                logloss_val = log_loss(y_val, pred_proba_val)

                val_rmse.append(rmse_val)
                val_ll.append(logloss_val)

    return train_rmse, train_ll, val_rmse, val_ll

        #tosave = np.array([ train_rmse, train_ll, val_rmse, val_ll ] )
        # np.save('colaborative_gridsearch', tosave)



    trainingData = ds.ix[train_ix][['step_id', 'student_id', 'correct_first_attempt']]
    valData = ds.ix[val_ix][['step_id', 'student_id', 'correct_first_attempt']]
    
    sf = gl.SFrame(trainingData)
    val_sf = gl.SFrame(valData)
    
    pred_proba_train = model.predict(sf)
    pred_proba_train = np.array(pred_proba_train)
    
    mse_train = mean_squared_error(y_train, pred_proba_train)
    rmse_train = np.sqrt(mse_train)
    logloss_train = log_loss(y_train, pred_proba_train)
    
    #Evaluation in validation set
    pred_proba_val = model.predict(val_sf)
    pred_proba_val = np.array(pred_proba_val)
    
    mse_val = mean_squared_error(y_val, pred_proba_val)
    rmse_val = np.sqrt(mse_val)
    logloss_val = log_loss(y_val, pred_proba_val)
    
    rmse_train
    rmse_val
    logloss_train
    logloss_val
    


