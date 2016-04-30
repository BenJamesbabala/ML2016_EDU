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

import re, string 
import copy
import csv

def cleanSubskills(SubskillsList):
    '''
    input: list of skills in array with no duplicates
    output: dictionary that maps bad input to good output
    Note: this is designed for Subskills
    '''
    newList=[]
    for skill in SubskillsList:
        if "[Skillrule" in skill:
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
        
    return newList2


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

    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    X = vectorizer.fit_transform(documents)
    model = KMeans(n_clusters = number_clusters , init='k-means++',
                    max_iter=100, n_init=1)
    model.fit(X)
    #order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    #terms = vectorizer.get_feature_names()
    return model.labels_

def remove_punctuation_from_docs(docs, replacewith=' '):
    ''' Remove punctuation from a string or a list of strings '''
    punctuation = '!"#$%&\'(),.:;<=>?@[\\]^_`{|}~'

    regex = re.compile('[%s]' % re.escape(punctuation))

    docs_mod = copy.deepcopy(docs)
    #If the parameter is only one word
    if isinstance(docs, str):
        return regex.sub(replacewith, docs)

    #Assume is a list of strings 
    for i in xrange(len(docs)):
        docs_mod[i] = regex.sub(replacewith, docs[i])

    return docs_mod

def tokenize_numbers_together(docs):
    ''' 
    docs: List of strings
    Change all numbers with the same token. 
    Returns: List of strings
    '''
    regex = re.compile('[0-9]+')
    docs_mod = copy.deepcopy(docs)
    #Assume is a list of strings 
    for i in xrange(len(docs)):
        docs_mod[i] = regex.sub('{N}', docs[i])

    return docs_mod

def lowercase_docs(docs):
    return [x.lower() for x in docs]

def remove_words_from_docs(docs, words ,replacewith=' '):
    ''' Remove words from a list of documents ''' 
    w_removed = copy.deepcopy(docs)
    for i in xrange(len(docs)):
        w_removed[i] = ' '.join(filter(lambda x: x.lower() not in words,
                                    docs[i].split()))
    return w_removed


def clusterDictionary(data, skillComponent, number_clusters =100, verbose=False):

    Subskills = data[skillComponent].apply(lambda x: str(x).split('~~'))
   
    # split lists of skills into individual skills
    SubskillsList = list(set(x for l in list(Subskills.values) for x in l))
   
   
    SubskillsListOrig = copy.deepcopy(SubskillsList)

    #clean subskills
    SubskillsList = cleanSubskills(SubskillsList)
    #Remove punctuation (verify what happens with math symbols)
    SubskillsList = remove_punctuation_from_docs(SubskillsList)
    #Lowercase all text
    SubskillsList = lowercase_docs(SubskillsList)
    #Remove words that do not add any information.
    SubskillsList = remove_words_from_docs(SubskillsList, ['skillrule'])
    #Tokenize numbers together
    SubskillsList = tokenize_numbers_together(SubskillsList)


    indexOfPartitions = kMeans(SubskillsList,number_clusters)
    partitions = getPartitions(SubskillsListOrig, indexOfPartitions)

    if verbose:
        for key in partitions.keys():
            print(key)
            for val in partitions[key]:
                print val

    return partitions


def main():

    fileLocation = '/home/michael/Desktop/machineLearningProject/27042016_train.txt' 
    data = pd.read_csv(fileLocation, sep ='\t',index_col=0)

    skill = 'kc_subskills'
    numberOfClusters = 100
    partitions = clusterDictionary(data, skill, numberOfClusters, verbose=True)
    
    s=0
    for key in partitions.keys():
        s+= len(partitions[key])

    
    print("number of keys")
    print(s)

    np.save(skill+str(numberOfClusters)+'.npy', partitions) 


def skills_cluster(data, skills_col, verbose=False):

    skills = data[skills_col].apply(lambda x: str(x).split('~~'))

    # split lists of skills into individual skills
    skillsListOrig = list(set(x for l in list(skills.values) for x in l))
    
    #Create a copy of the subskills to clean and cluster with
    skillsList = copy.deepcopy(skillsListOrig)
    #clean subskills
    skillsList = cleanSubskills(skillsList).values()
    #Remove punctuation (verify what happens with math symbols)
    skillsList = remove_punctuation_from_docs(skillsList)
    #Lowercase all text
    skillsList = lowercase_docs(skillsList)
    #Remove words that do not add any information.
    skillsList = remove_words_from_docs(skillsList, ['skillrule'])
    #Tokenize numbers together
    skillsList = tokenize_numbers_together(skillsList)

    #Similarity not needed for now
    #subSkillSimilarities = cosSimilarity(skillsList)

    indexOfPartitions = kMeans(skillsList, 40)
    partitions = getPartitions(skillsList, indexOfPartitions)
    partitions_orig = getPartitions(skillsListOrig, indexOfPartitions)
    
    if verbose:
        for key in partitions_orig.keys():
            print(key)
            for val in partitions_orig[key]:
                print val





if __name__ == '__main__':
    main()


