import pandas as pd 
import numpy as np 
import graphlab as gl


def getLatents(trainingData, user = 'student_id', item = 'step_id',targ = 'correct_first_attempt', allVariables = False,user_data = None, item_data=None ,num_factors = 8 ,max_iterations =100):
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
                                                            verbose =True)
    
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
    data = ds
    itemDF,userDF, model = getLatents(data, user = 'student_id', 
                                        item = 'step_id',targ = 'correct_first_attempt',
                                        allVariables = False, num_factors = 4)

    #print model['coefficients']
    #print( itemDF.shape )
    
    #factorsToMergeWithData(data, itemDF, userDF).to_csv('latentCombinations.csv' )





if __name__ == '__main__':
        main()





#testData = ds.ix[test_ix][['step_id', 'student_id']]
#sf_test = gl.SFrame(testData)
#model.predict(testData)
#
#y_test = ds.ix[test_ix].correct_first_attempt
#np.sqrt(mean_squared_error(y_test, pred))
#
#merged1[merged1.step_id == '0REAL20BR3C1']
#rged
#
#merged2[merged2.student_id == 'stu_8575574503']
#
#
#stu_244b6a9fdd