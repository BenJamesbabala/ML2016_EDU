import numpy as np
from scipy.sparse import csr_matrix


def dictionary_reverser( skillsDictionary):
	tuplesOfSkills=[]
	for key in skillsDictionary.keys():
	    for value in skillsDictionary[key]:
	        tuplesOfSkills.append((value,key))
	return dict(tuplesOfSkills)



def lookUpDictionary(subskills_vectorizer, skillsDictionary):
	reverseSkillsDictionary = dictionary_reverser(skillsDictionary)
	indexedTuples=[]
	for skill in reverseSkillsDictionary.keys():
	    ind = features.index(skill)
	    indexedTuples.append((ind,reverseSkillsDictionary[skill]))

	a = dict(indexedTuples)

	final = dict()
	for i in range(len(skillsDictionary)):
	    final[i]=[]
	    
	for key in a.keys():
	    final[a[key]].append(key)

	return final

def sparse_matrix_clusterer(sparse_matrix, clusters_dict):
    
    dummyMatrix=csr_matrix((sparse_matrix.shape[0],len(clusters_dict)))
    for key in clusters_dict.keys():
        skills_list = clusters_dict[key]
        dummyMatrix[:,int(key)] = sparse_matrix[:,skills_list].sum(axis=1)
        
    return dummyMatrix
