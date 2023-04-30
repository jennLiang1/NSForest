### Libraries ###
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import itertools
import time
import os

### My functions ###

# class_weight_cal1 calculates class weight for weight_scheme = 'class_weight' option of NSForest function
# class_weight_cal2 calculates class weight for weight_scheme = 'binary' option of NSForest function

def class_weight_cal1(df_dummies, weight, cl):  
    cl_list = df_dummies.columns.to_list()
    cl_len = len(cl_list)
    keys = range(cl_len)
    
    if cl == 'Inh L1-2 PAX6 CDH12':
        level1 = weight.xs(cl, level = 1, drop_level=False)
        weight_cl = level1.swaplevel(i=-2, j=-1)
    elif cl == 'Micro L1-6 TYROBP':
        weight_cl = weight.loc[[cl]]
    else:
        level0 = weight.loc[[cl]]
        level1 = weight.xs(cl, level = 1, drop_level=False)
        level1 = level1.swaplevel(i=-2, j=-1) #swap index name for consistency
        weight_cl = pd.concat([level0, level1])

    cl_weight = weight_cl.loc[cl]
    cl_weight.loc[len(cl_weight.index)] = [min(cl_weight.iloc[:, 0])]# add the lowest weight as the weight to self
    cl_weight = cl_weight.rename(index = {len(cl_weight.index)-1: 'Inh L1-2 PAX6 CDH12'})
    sorted_weight = [cl_weight[cl_weight.index == i] for i in cl_list]
    matched_weight = pd.concat(sorted_weight)
    
    values = matched_weight.values
    class_weight_cl = dict(zip(keys, values))
    
    return class_weight_cl

def class_weight_cal2(df_dummies, weight, cl):  
    cl_list = df_dummies.columns.to_list()
    cl_len = len(cl_list)
    if cl == 'Inh L1-2 PAX6 CDH12':
        level1 = weight.xs(cl, level = 1, drop_level=False)
        weight_cl = level1.swaplevel(i=-2, j=-1)
    elif cl == 'Micro L1-6 TYROBP':
        weight_cl = weight.loc[[cl]]
    else:
        level0 = weight.loc[[cl]]
        level1 = weight.xs(cl, level = 1, drop_level=False)
        level1 = level1.swaplevel(i=-2, j=-1) #swap index name for consistency
        weight_cl = pd.concat([level0, level1])
    
    cl_weight = weight_cl.loc[cl]
    cl_weight.loc[len(cl_weight.index)] = [min(cl_weight.iloc[:, 0])]# add the lowest weight as the weight to self
    cl_weight = cl_weight.rename(index = {len(cl_weight.index)-1: 'Inh L1-2 PAX6 CDH12'})
    sorted_weight = [cl_weight[cl_weight.index == i] for i in cl_list]
    match_weight = pd.concat(sorted_weight)
    
    keys = cl_list
    values = match_weight.values
    class_weight_cl = dict(zip(keys, values)) 
    
    return class_weight_cl

## run Random Forest on the binary dummy variables ==> outputs all genes ranked by Gini impurit
def myRandomForest1(adata, df_dummies, cl, n_trees, n_jobs, n_top_genes, class_weight_cl):
    x_train = adata.X
    y_train = df_dummies[cl]
    if class_weight_cl == None:
        rf_clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs, random_state=123456)
    elif class_weight_cl == class_weight_cl: 
        rf_clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs, random_state=123456, class_weight = class_weight_cl) #<===== criterion=“gini”, by default
    rf_clf.fit(x_train, y_train)
    ## get feature importance and rank/subset top genes
    top_rf_genes = pd.Series(rf_clf.feature_importances_, index=adata.var_names).sort_values(ascending=False)[:n_top_genes]
    return top_rf_genes    

## construct decision tree for each gene and evaluate the fbeta score in all combinations ==> outputs markers with max fbeta, and all scores
def myDecisionTreeEvaluation1(adata, df_dummies, cl, genes_eval, beta, class_weight_cl):
    dict_pred = {}
    for i in genes_eval:
        x_train = adata[:,i].X
        y_train = df_dummies[cl]
        if class_weight_cl == None:
            tree_clf = DecisionTreeClassifier(max_leaf_nodes=2)
        elif class_weight_cl == class_weight_cl: 
            tree_clf = DecisionTreeClassifier(max_leaf_nodes=2, class_weight = class_weight_cl)
        tree_clf = tree_clf.fit(x_train, y_train) 
        dict_pred[i] = tree_clf.apply(x_train)-1
    df_pred = pd.DataFrame(dict_pred)
    
    combs = []# gene combination
    for L in range(1, len(genes_eval)+1):
        els = [list(x) for x in itertools.combinations(genes_eval, L)]
        combs.extend(els)
    
    dict_scores = {} 
    for ii in combs:
        y_true = df_dummies[cl]
        y_pred = df_pred[ii].product(axis=1)
        fbeta = fbeta_score(y_true, y_pred, average='binary', beta=beta)
        ppv = precision_score(y_true, y_pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        dict_scores['&'.join(ii)] = fbeta, ppv, tn, fp, fn, tp
    df_scores = pd.DataFrame(dict_scores)
        
    ## find which combination has the max fbeta
    idx_max = df_scores.idxmax(axis=1)[0] #[0] is fbeta
    markers = idx_max.split('&')
    scores = df_scores[idx_max]
    score_max = scores[0]
    return markers, scores, score_max

###################
## Main function ##
###################

def NSForest_weighted(adata, cluster_header, weight, weight_scheme, cluster_list=None, medians_header=None, 
             n_trees=1000, n_jobs=-1, beta=0.5, n_top_genes=15, n_binary_genes=10, n_genes_eval=6,
             output_folder="NSForest_outputs/"):
    
    ## set up outpu folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("Preparing data...")
    start_time = time.time()
    ## densify X from sparse matrix format
    adata.X = adata.to_df()
    ## categorial cluster labels
    adata.obs[cluster_header] = adata.obs[cluster_header].astype('category')   
    # dummy/indicator for one vs. all Random Forest model    
    df_dummies = pd.get_dummies(adata.obs[cluster_header]) #cell-by-cluster
    
    
    ## get number of cluster
    n_total_clusters = len(df_dummies.columns)
    print("--- %s seconds ---" % (time.time() - start_time))

    if medians_header == None:
        print("Calculating medians...")
        start_time = time.time()
        ## get dataframes for X and cluster in a column
        df_X = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names) #cell-by-gene
        clusters = adata.obs[cluster_header]
        df_X_clusters = pd.concat([df_X, clusters], axis=1)
        ## get cluster medians
        cluster_medians = df_X_clusters.groupby([cluster_header]).median() #cluster-by-gene
        
        ## delete to free up memories
        del df_X, clusters, df_X_clusters
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print("Getting pre-calculated medians...")
        start_time = time.time()
        cluster_medians = adata.varm[medians_header].transpose() #cluster-by-gene
        print("--- %s seconds ---" % (time.time() - start_time))
    
    ### START iterations ###
    if cluster_list == None:
        cluster_list = df_dummies.columns
    n_clusters = len(cluster_list)
    
    print ("Number of clusters to evaluate: " + str(n_clusters))
    ct = 0
    df_supp = df_markers = df_results = pd.DataFrame()
    start_time = time.time()
    for cl in cluster_list:
        ct+=1
        print(str(ct) + " out of " + str(n_clusters) + ":")

        ## cluster in iteration
        print("\t" + cl)
        
        ##=== reset parameters for this iteration!!! (for taking care of special cases) ===##
        n_binary_genes_cl = n_binary_genes
        n_genes_eval_cl = n_genes_eval
        
        ########################################## [ADDED] #########################################
        if weight_scheme == 'class_weight':
            class_weight_cl = class_weight_cal1(df_dummies, weight, cl)
        elif weight_scheme == 'binary':
            class_weight_cl = class_weight_cal2(df_dummies, weight, cl)    
        
        ## Random Forest step: get top genes ranked by Gini/feature importance
        if weight_scheme == 'class_weight':
            top_rf_genes = myRandomForest1(adata, df_dummies, cl, n_trees, n_jobs, n_top_genes, class_weight_cl)
        elif weight_scheme == 'binary':
            top_rf_genes = myRandomForest1(adata, df_dummies, cl, n_trees, n_jobs, n_top_genes, class_weight_cl = None)
        ########################################## [MODIFIED] #########################################

        ## filter out negative genes by thresholding median>0 ==> to prevent dividing by 0 in binary score calculation
        top_gene_medians = cluster_medians.loc[cl,top_rf_genes.index]
        top_rf_genes_positive = top_gene_medians[top_gene_medians>0]
        n_positive_genes = sum(top_gene_medians>0)
    
        ##=== special cases: ===##
        if n_positive_genes == 0:
            print("\t" + "No positive genes for evaluation. Skipped. Optionally, consider increasing n_top_genes.")
            continue

        if n_positive_genes < n_binary_genes:
            print("\t" + f"Only {n_positive_genes} out of {n_top_genes} top Random Forest features with median > 0 will be further evaluated.")
            n_binary_genes_cl = n_positive_genes
            n_genes_eval_cl = min(n_positive_genes, n_genes_eval)
        ##===##
        
        ########################################## [MODIFIED] #########################################
        ## Binary scoring step: calculate binary scores for all positive top genes
        if weight_scheme == 'class_weight': 
            binary_scores = [sum(np.maximum(0,1-cluster_medians[i]/cluster_medians.loc[cl,i]))/(n_total_clusters-1) for i in top_rf_genes_positive.index]
        elif weight_scheme == 'binary': 
            cl_order = list(class_weight_cl.keys())
            cl_medians_ordered = cluster_medians.loc[cl_order]
            binary_scores = [sum(np.maximum(0,1-cluster_medians[i]/cl_medians_ordered.loc[cl,i]) 
                               .dot(list(class_weight_cl.values())))/(len(class_weight_cl.values())-1) for i in top_rf_genes_positive.index]
        ########################################## [MODIFIED] #########################################
        
        top_binary_genes = pd.Series(binary_scores, index=top_rf_genes_positive.index).sort_values(ascending=False)

        ## Evaluation step: calculate F-beta score for gene combinations
        genes_eval = top_binary_genes.index[:n_genes_eval_cl].to_list()
        ########################################## [MODIFIED] #########################################
        if weight_scheme == 'class_weight':
            markers, scores, score_max = myDecisionTreeEvaluation1(adata, df_dummies, cl, genes_eval, beta, class_weight_cl)
        elif weight_scheme == 'binary':
            markers, scores, score_max = myDecisionTreeEvaluation1(adata, df_dummies, cl, genes_eval, beta, class_weight_cl = None)
        ########################################## [MODIFIED] #########################################
                
        print("\t" + str(markers))
        print("\t" + str(score_max))

        ## return supplementary table as csv
        binary_genes_list = top_binary_genes.index[:n_binary_genes_cl].to_list()
        df_supp_cl = pd.DataFrame({'clusterName': cl,
                                   'binary_genes': binary_genes_list,
                                   'rf_feature_importance': top_rf_genes[binary_genes_list],
                                   'cluster_median': top_gene_medians[binary_genes_list],
                                   'binary_score': top_binary_genes[binary_genes_list]}).sort_values('binary_score', ascending=False)
        df_supp = pd.concat([df_supp,df_supp_cl]).reset_index(drop=True)
        df_supp.to_csv(output_folder + "NSForest_supplementary1.csv", index=False)

        ## return markers table as csv
        df_markers_cl = pd.DataFrame({'clusterName': cl, 'markerGene': markers, 'score': scores[0]})
        df_markers = pd.concat([df_markers, df_markers_cl]).reset_index(drop=True)
        df_markers.to_csv(output_folder + "NSForest_markers1.csv", index=False)

        ## return final results as dataframe
        dict_results_cl = {'clusterName': cl,
                           'f_score': scores[0],
                           'PPV': scores[1],
                           'TN': int(scores[2]),
                           'FP': int(scores[3]),
                           'FN': int(scores[4]),
                           'TP': int(scores[5]),
                           'marker_count': len(markers),
                           'NSForest_markers': [markers],
                           'binary_genes': [df_supp_cl['binary_genes'].to_list()] #for this order is the same as the supp order
                           }
        df_results_cl = pd.DataFrame(dict_results_cl)
        df_results = pd.concat([df_results,df_results_cl]).reset_index(drop=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    ### END iterations ###

    return(df_results)
