from sklearn.cluster import KMeans


#def cluster_from_performance(ds, skills_sparse_cl):









def main():
    problem_latents = latent[[u'itemLatent0', u'itemLatent1', 
                        u'itemLatent2', u'itemLatent3',
                        u'itemLatent4', u'itemLatent5', 
                        u'itemLatent6', u'itemLatent7']]

    student_latents = latent[[u'userLatent0', u'userLatent1', 
                            u'userLatent2', u'userLatent3',
                            u'userLatent4', u'userLatent5', 
                            u'userLatent6', u'userLatent7']]

    clusterer = KMeans(n_clusters=20, init='k-means++', 
                        n_init=10, max_iter=300, tol=0.0001, 
                        precompute_distances='auto', verbose=0, 
                        random_state=None, copy_x=True, n_jobs=15)

    p_clusters = clusterer.fit_predict(problem_latents)

    s_clusters = clusterer.fit_predict(student_latents)
    
    X_ds['s_k'] = s_clusters
    X_ds['p_k'] = p_clusters
    to_dummy = ['s_k', 'p_k']
    X_ds_d = pd.get_dummies(X_ds, columns=to_dummy)




if __name__ == '__main__':
    main()



#Index([u'itemLatent0', u'itemLatent1', u'itemLatent2', u'itemLatent3',
#       u'itemLatent4', u'itemLatent5', u'itemLatent6', u'itemLatent7',
#       u'userLatent0', u'userLatent1', u'userLatent2', u'userLatent3',
#       u'userLatent4', u'userLatent5', u'userLatent6', u'userLatent7',
#       u'cross0', u'cross1', u'cross2', u'cross3', u'cross4', u'cross5',
#       u'cross6', u'cross7'],
#      dtype='object')
