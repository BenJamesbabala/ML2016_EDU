import pandas as pd 
import numpy as np 
import graphlab as gl


def getLatents(trainingData, user = 'student_id', item = 'step_id',targ = 'correct_first_attempt', allVariables = False,user_data = None, item_data=None):
    '''
    input: pandas dataframe, need columns names for user/item uses default student_id and step_id
        CAN provide pandas df that summarizes users and or items
    output: two pandas dataframe and the model that produces them- one df for users one df for items, holds latent variables and linear term
    model has .predict() function that you can feed a df with same format into, will return df of scores and ranking for problems/students
    '''


    itemDF = pd.DataFrame()
    userDF = pd.DataFrame()


    if user_data !=None:
        user_data = gl.SFrame(user_data)
    if item_data !=None:
        user_data = gl.SFrame(item_data)

    if allVariables==False:
        #dont use extra data
        trainingData = trainingData[[item,user,targ]]

    sf = gl.SFrame(trainingData)
    model =gl.recommender.create( sf , user_id= user,item_id = item ,target = targ, ranking =False, user_data=user_data, item_data=item_data)
    recomendations= model.recommend()

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
	'''
	itemFactors and userFactors are output from getLatents, data is data that getLatents was apllied to
	output: pandas DF whose index are same as data and matches


	'''
	data= data[['student_id','step_id']]
	final = pd.DataFrame()
	merged1 = data.merge( itemFactors, how = 'inner',left_on='step_id', right_on='step_id')
	merged2 = userFactors.merge(merged1,how = 'inner',left_on='student_id',right_on='student_id')
        
	for i in range(itemFactors.shape[1]-1):
		merged2['cross'+str(i)] = merged2[str('itemLatent'+ str(i))] * merged2[str('userLatent'+ str(i))]  
        
	merged2.drop(['student_id','step_id'],inplace=True,axis=1)
    
	return merged2



def main():
	location = '/home/michael/Desktop/machineLearningProject/27042016_train.txt' 
	data = pd.read_csv(location,sep ='\t',nrows =100000 ,skipinitialspace =False)

	itemDF,userDF, model = getLatents(data, user = 'student_id', item = 'step_id',targ = 'correct_first_attempt', allVariables = False)

	#print model['coefficients']
	#print( itemDF.shape )
	print factorsToMergeWithData(data, itemDF, userDF).head()
	




if __name__ == '__main__':
    	main()

