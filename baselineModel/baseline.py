import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


data = pd.read_csv('/home/michael/Desktop/machineLearningProject/algebra_2008_2009/algebra_2008_2009_train.txt' , delimiter ='\t')

data['fullStepName']= data['Problem Name'] +data['Step Name'] 
cleanData = data[['Anon Student Id', 'fullStepName', 'Correct First Attempt']]


difficultyByProblem = cleanData.groupby('fullStepName')['Correct First Attempt'].mean()
deviationByProblem = cleanData.groupby('fullStepName')['Correct First Attempt'].std()
correctByStudent = cleanData.groupby('Anon Student Id')['Correct First Attempt'].mean()
deviationByStudent = cleanData.groupby('Anon Student Id')['Correct First Attempt'].std()

studentIds = list(cleanData['Anon Student Id'].unique())[:200]
problemNames = list(cleanData['fullStepName'].values)



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def objectiveFunction(df,theta,beta):
	'''
	theta must be a series with the studentId as key
	beta must be a series with problemName as key
	returns float
	iterates over every row in df

	'''
   
    total = 0
    
    
    studentPriorMean = 0
    studentPriorStd = 1
    problemPriorMean = 0
    problemPriorStd = 1
    
    for row in df.iterrows():        
        studentId =  row[1][0]
        correct = row[1][2]
        problemName = row[1][1]
        
        
        
        priorStudent = -(theta[studentId] - studentPriorMean )**2/(2*studentPriorStd**2)
        priorProblem = -(beta[problemName] - problemPriorMean )**2/(2*problemPriorStd**2)
        
        
        if correct:
            joint = np.log( sigmoid(  theta[studentId] - beta[problemName]  ) )
        else:
            joint = np.log(1  - sigmoid( theta[studentId] - beta[problemName] ))
        
        total += priorStudent + priorProblem + joint
    
    return total
            