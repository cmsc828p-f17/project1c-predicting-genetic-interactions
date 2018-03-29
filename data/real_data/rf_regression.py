import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import scipy
from sklearn.model_selection import KFold
import os
from collections import defaultdict
import pickle
from sklearn.metrics import r2_score

# Set random seed
np.random.seed(0)

# read genes and terms information
def reader():
    df = pd.read_csv('/input/gene_term.csv')
    genes = df['Gene'].tolist()
    terms = df['Term'].tolist()
    num_records = len(genes)

    # create term_dict which contains genes in that term
    term_dict = {}
    for i in range(num_records):
        term = terms[i]
        if term not in term_dict:
            term_dict[term] = [genes[i]]
        else:
            term_dict[term].append(genes[i])

    term_set = list(term_dict.values())
    return genes, terms, term_set



def obtain_interactions():
    interactions = []
    with open('/input/collins-sc-emap-gis.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            interactions.append(row)

    interactions = interactions[1:]
    # num_inter_records = len(interactions)    # 150636
    # print("number of interactions: ", num_inter_records)
    gene1 = [inter[0] for inter in interactions]
    gene2 = [inter[1] for inter in interactions]
    interaction_genes = sorted(list(set(gene1 + gene2)))
    # build interaction matrix
    gene_pairs = []
    pair_scores = []
    interactions_matrix = defaultdict(dict)
    for inter in interactions:
        interactions_matrix[inter[0]][inter[1]] = inter[2]
        gene_pairs.append((inter[0],inter[1]))
        pair_scores.append(inter[2])
    return interaction_genes,interactions_matrix, gene_pairs,pair_scores


def build_geneterm_dict(genes, term_set):
    # build gene dict which returns the terms that specific gene belongs
    gene_dict = {}
    for gid,gene in enumerate(genes):
        for idx, term in enumerate(term_set):
            if gene in term:
                if gene not in gene_dict:
                    gene_dict[gene] = [idx]
                else:
                    gene_dict[gene].append(idx)
    # print(len(gene_dict.items()))   # 653
    return gene_dict

def construct_feature_vector(gene_pairs, gene_dict, pair_scores, num_terms):
    feature_vector = np.zeros([len(gene_pairs), num_terms+1])
    # only interaction pairs (num_interactions) x (5125+1)  assume (g1,g2) is equivalent to (g2,g1)
    row_idx = 0
    for idx,(gene1,gene2) in enumerate(gene_pairs):
        if gene1 in gene_dict and gene2 in gene_dict:
            terms_i = gene_dict[gene1]
            terms_j = gene_dict[gene2]
            feature_vector[row_idx][terms_i] += 1
            feature_vector[row_idx][terms_j] += 1
            feature_vector[row_idx][-1] = pair_scores[idx]
            row_idx +=1
    return feature_vector[0:row_idx,:]               # size of 146566x 5216


def label_interaction(preds):
    labels = []
    for pred in preds:
        if pred < -0.08:
            labels.append(1)#('negative')
        elif pred > 0.08:
            labels.append(2)#('positive')
        else:
            labels.append(0)#('no-interaction')
    return labels

def CrossValidation(X,y,num_split):
    # make four fold cross validation
    k_fold = KFold(n_splits=num_split,shuffle=True)
    k_fold.get_n_splits(X)

    # create random forest regressor
    regr = RandomForestRegressor(min_samples_split=3)
    # fit on the training set
    all_correlations= []
    for k, (train, test) in enumerate(k_fold.split(X)):
        print("training size:",len(train),"testing size:",len(test))
        regr.fit(X[train], y[train])
        pred_score = regr.predict(X[test])
        true_score = y[test]
        # calculate correlation
        correlation = scipy.stats.pearsonr(true_score, pred_score)
        print("For fold: ", k, ", the correlations is:", correlation[0])#,r2_score(true_score,pred_score))
        all_correlations.append(correlation[0])
        # label the interactions from the scores
        preds_label = label_interaction(pred_score)
        true_label = label_interaction(true_score)
        # generate confusion matrix
        confusion_matrix = pd.crosstab(pd.Series(true_label), pd.Series(preds_label), rownames=['Actual interaction'], colnames=['Predicted interaction'])
        print(confusion_matrix,"\n")

        # save scores and indexes
        pickle.dump([pred_score,true_score,correlation], open("/output/{0}th_result.pickle".format(k), "wb"))
        pickle.dump([train,test],open("/output/{0}th_indexes.pickle".format(k), "wb"))
    print("final correlation is:", max(all_correlations))


if __name__ == "__main__":

    # read data
    genes, terms, term_set = reader()
    num_terms = len(term_set)     # 5125

    # obtain interactions information
    interaction_genes, interactions_matrix,gene_pairs,pair_scores = obtain_interactions()

    # ordering the interaction_matrix
    # print("odering interaction matrix...")
    # ordered_interaction_matrix = ordering(interactions_matrix, interaction_genes)

    # help to construct feature vector
    print("building gene_dict...")
    gene_dict = build_geneterm_dict(interaction_genes, term_set)

    print("gene_dict done! Now pairing genes and scores...")
    # gene_pairs, pair_scores = pair_genes(interaction_genes, ordered_interaction_matrix)

    print("paired ",len(gene_pairs),"genes! Now constructing feature vectors...")
    # construct feature vector only for gene pairs with genetic interaction
    feature_vector = construct_feature_vector(gene_pairs,gene_dict, pair_scores, num_terms)
    feature_vector = np.asarray(feature_vector)   # (664*663/2)x5126
    print(feature_vector.shape)

    # separate into features and scores
    X = feature_vector[:,:-1]
    y = feature_vector[:,-1]

    print("start training...")
    #four-fold cross validation
    CrossValidation(X,y,4)

# ################################################################################
#
# 2017-10-19 14:41:09,115 INFO - Run Output:
# 2017-10-19 14:41:10,421 INFO - building gene_dict...
# 2017-10-19 14:41:14,767 INFO - gene_dict done! Now pairing genes and scores...
# 2017-10-19 14:41:14,769 INFO - paired  150636 genes! Now constructing feature vectors...
# 2017-10-19 14:41:21,002 INFO - (146566, 5126)
# 2017-10-19 14:41:21,003 INFO - start training...
# 2017-10-19 14:41:21,007 INFO - training size: 109924 testing size: 36642
# 2017-10-19 14:56:30,911 INFO - For fold:  0 , the correlations is: 0.474094159198
# 2017-10-19 14:56:30,975 INFO - Predicted interaction     0     1     2
# 2017-10-19 14:56:30,975 INFO - Actual interaction
# 2017-10-19 14:56:30,975 INFO - 0                       436  1060  1240
# 2017-10-19 14:56:30,975 INFO - 1                      1973  8211  5544
# 2017-10-19 14:56:30,975 INFO - 2                      2572  6522  9084
# 2017-10-19 14:56:30,976 INFO -
# 2017-10-19 14:56:31,087 INFO - training size: 109924 testing size: 36642
# 2017-10-19 15:12:07,558 INFO - For fold:  1 , the correlations is: 0.48556516452
# 2017-10-19 15:12:07,607 INFO - Predicted interaction     0     1     2
# 2017-10-19 15:12:07,607 INFO - Actual interaction
# 2017-10-19 15:12:07,607 INFO - 0                       421  1113  1231
# 2017-10-19 15:12:07,607 INFO - 1                      1989  8126  5421
# 2017-10-19 15:12:07,607 INFO - 2                      2597  6751  8993
# 2017-10-19 15:12:07,607 INFO -
# 2017-10-19 15:12:07,725 INFO - training size: 109925 testing size: 36641
# 2017-10-19 15:27:40,189 INFO - For fold:  2 , the correlations is: 0.474034305647
# 2017-10-19 15:27:40,238 INFO - Predicted interaction     0     1     2
# 2017-10-19 15:27:40,239 INFO - Actual interaction
# 2017-10-19 15:27:40,239 INFO - 0                       428  1080  1194
# 2017-10-19 15:27:40,239 INFO - 1                      1942  8087  5543
# 2017-10-19 15:27:40,239 INFO - 2                      2448  6877  9042
# 2017-10-19 15:27:40,239 INFO -
# 2017-10-19 15:27:40,352 INFO - training size: 109925 testing size: 36641
# 2017-10-19 15:44:09,191 INFO - For fold:  3 , the correlations is: 0.483916062072
# 2017-10-19 15:44:09,239 INFO - Predicted interaction     0     1     2
# 2017-10-19 15:44:09,239 INFO - Actual interaction
# 2017-10-19 15:44:09,240 INFO - 0                       398  1065  1291
# 2017-10-19 15:44:09,240 INFO - 1                      1936  8232  5523
# 2017-10-19 15:44:09,240 INFO - 2                      2420  6704  9072
# 2017-10-19 15:44:09,240 INFO -
# 2017-10-19 15:44:09,363 INFO - final correlation is: 0.48556516452
# 2017-10-19 15:44:09,574 INFO -
# ################################################################################



