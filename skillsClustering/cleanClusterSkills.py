import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def cleanSubskills(SubskillsList):
    '''
    input: list of skills in array with no duplicates
    output: dictionary that maps bad input to good output
    Note: this is designed for Subskills
    '''
    newList=[]
    for skill in SubskillsList:
        if "{" in skill:
            splittedField = skill.split("{")[0][12:-2]
            splittedField2 = splittedField.split('[')
            newList.append(splittedField2[0])
        else:

            newList.append(skill)
    newList2=[]
    for skill in newList:
        if 'SkillRule' in skill:
            skill = skill.split(';')[0][12:]
        newList2.append(skill)
        

    return dict(zip(SubskillsList,newList))


def cosSimilarity(documents):
    '''
    documents is an array of words 
    output is pandas df with entry i,j corresponding to doc frequency of doc i and doc j
    in documents 
    *term frequency is taken into account
    '''
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)





def affinityCluster(similarityMatrix , param = -10):
	'''
	input: nxn matrix where the i,j the entrity gives the similarity between 
	document i and document j
	output: numpy array where ith entry gives cluster document i falls into

	'''
	af = AffinityPropagation(preference= param).fit(similarityMatrix)
	cluster_centers_indices = af.cluster_centers_indices_
	return af.labels_ 

def getPartitions(listOfSkills, listOfPartitions):
	'''
	output: dictionary where key (0,1,...,) is cluster that items fall into according to cluster
	algorithm

	'''
	
	groupingDictionary = dict()
	for i in range( listOfPartitions.max() +1 ):
	    groupingDictionary[i]=[]  # set key to be integers 0,1,..., numPartitions
	    
	for i in range(len(listOfSkills)):
	    groupingDictionary[listOfPartitions[i]].append(listOfSkills[i])

	return groupingDictionary


def kMeans(documents , number_clusters):
    '''
    documents is a list of Strings
    output is a list with the same length as documents, where each entry is a number 0,1,...,number_clusters

    '''

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    model = KMeans(n_clusters = number_clusters , init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
	#order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	#terms = vectorizer.get_feature_names()
    return model.labels_

def main():

	fileLocation = '/home/michael/Desktop/machineLearningProject/ML2016_EDU/Datasets/22042016_train.txt' 
	data = pd.read_csv(fileLocation, sep ='\t',index_col=0)


	Subskills = data['kc_subskills'].apply(lambda x: str(x).split('~~'))
	tracedSkills = data['k_traced_skills'].apply(lambda x: str(x).split('~~'))
	rules =data['kc_rules'].apply(lambda x: str(x).split('~~'))

	# split lists of skills into individual skills
	SubskillsList = list(set(x for l in list(Subskills.values) for x in l))
	tracedSkillsList = list(set(x for l in list(tracedSkills.values) for x in l))
	rulesList = list(set(x for l in list(rules.values) for x in l))

	#clean subskills
	SubskillsList = cleanSubskills(SubskillsList).values()


	subSkillSimilarities = cosSimilarity(SubskillsList)
	tracedSkillsSimilarities = cosSimilarity(tracedSkillsList)
	rulesSimilarities = cosSimilarity(rulesList)

	indexOfPartitions = kMeans(SubskillsList, 100)
	partitions = getPartitions(SubskillsList, indexOfPartitions)

	for key in partitions.keys():
		print(key)
		for val in partitions[key]:
			print val



if __name__ == '__main__':
    main()
