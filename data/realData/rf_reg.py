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
    # num_inter_records = len(interactions)
    # print("number of interactions: ", num_inter_records)

    # obtain genes in interactions
    gene1 = [inter[0] for inter in interactions]
    gene2 = [inter[1] for inter in interactions]
    interaction_genes = sorted(list(set(gene1+gene2)))   # 664

    # build interaction matrix
    interactions_matrix = defaultdict(dict)
    for inter in interactions:
        interactions_matrix[inter[0]][inter[1]] = inter[2]
    return interaction_genes, interactions_matrix

# ordering the matrix
def ordering(interactions_matrix,interaction_genes):
    ordered_interaction_matrix = defaultdict(dict)
    for gene1 in interaction_genes:
      for gene2 in interaction_genes:
        if gene1 in interactions_matrix.keys() and gene2 in interactions_matrix[gene1].keys():
          ordered_interaction_matrix[gene1][gene2] = interactions_matrix[gene1][gene2]
        else:
          ordered_interaction_matrix[gene1][gene2] = 0
    return ordered_interaction_matrix

def pair_genes(genes,ordered_interactions_matrix):
    # produce gene interaction pairs
    gene_pairs = []
    pair_scores = []
    # class_dict = {"positive" : 1, "negative":2,"non-interaction":0}
    num_genes = len(genes)

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gene_pairs.append((genes[i], genes[j]))
            pair_scores.append(ordered_interaction_matrix[genes[i]][genes[j]])
    return gene_pairs, pair_scores

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
        # if gid%400 == 0:
        #     print(int(gid/len(genes)*100),"% finished...")
    return gene_dict

def construct_feature_vector(gene_pairs, gene_dict, pair_scores, num_terms):
    feature_vector = np.zeros([len(gene_pairs), num_terms+1])
    # only interaction pairs(664x663/2) x (5125+1)  4950 gene combination(assume (g1,g2) is equivalent to (g2,g1))
    row_idx = 0
    for idx,(gene1,gene2) in enumerate(gene_pairs):
        if gene1 in gene_dict and gene2 in gene_dict:
            terms_i = gene_dict[gene1]
            terms_j = gene_dict[gene2]
            feature_vector[row_idx][terms_i] += 1
            feature_vector[row_idx][terms_j] += 1
            feature_vector[row_idx][-1] = pair_scores[idx]
            row_idx +=1

        # if idx%500 == 0:
        #     print(int(idx/len(gene_pairs)*100),"% finished...")
    return feature_vector


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
    k_fold = KFold(n_splits=num_split)
    k_fold.get_n_splits(X)

    # create random forest regressor
    regr = RandomForestRegressor(max_depth=None, random_state=0)
    # fit on the training set
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        print("training size:",len(train),"testing size:",len(test))
        regr.fit(X[train], y[train])
        pred_score = regr.predict(X[test])
        true_score = y[test]
        # calculate correlation
        correlation = scipy.stats.pearsonr(true_score, pred_score)
        print("For fold: ", k, ", the correlations is:", correlation[0])#,r2_score(true_score,pred_score))

        # label the interactions from the scores
        preds_label = label_interaction(pred_score)
        true_label = label_interaction(true_score)
        # generate confusion matrix
        confusion_matrix = pd.crosstab(pd.Series(true_label), pd.Series(preds_label), rownames=['Actual interaction'], colnames=['Predicted interaction'])
        print(confusion_matrix,"\n")

        # save scores and indexes
        pickle.dump([pred_score,true_score,correlation], open("/output/{0}th_result.pickle".format(k), "wb"))
        pickle.dump([train,test],open("/output/{0}th_indexes.pickle".format(k), "wb"))


if __name__ == "__main__":

    # read data
    genes, terms, term_set = reader()
    num_terms = len(term_set)     # 5125

    # obtain unique genes
    unique_genes = list(set(genes))  # 6406

    # obtain interactions information
    interaction_genes, interactions_matrix = obtain_interactions()

    # ordering the interaction_matrix
    print("odering interaction matrix...")
    ordered_interaction_matrix = ordering(interactions_matrix, interaction_genes)

    # help to construct feature vector
    print("building gene_dict...")
    gene_dict = build_geneterm_dict(unique_genes, term_set)

    print("gene_dict done! Now pairing genes and scores...")
    gene_pairs, pair_scores = pair_genes(interaction_genes, ordered_interaction_matrix)

    print("pairing finished! Now constructing feature vectors...")
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


############################ Results############################################

# training size: 165087 testing size: 55029
# For fold:  0 , the correlations is: 0.328052010826
# Predicted interaction     0     1     2
# Actual interaction
# 0                      4795  7569  7175
# 1                      2492  8540  5345
# 2                      3183  7776  8154
#
# training size: 165087 testing size: 55029
# For fold:  1 , the correlations is: 0.428553727311
# Predicted interaction     0     1     2
# Actual interaction
# 0                      5896  6849  7390
# 1                      2657  7933  5726
# 2                      3133  6927  8518
#
# 165087 testing size: 55029
# For fold:  2 , the correlations is: 0.336450787305
# Predicted interaction     0     1     2
# Actual interaction
# 0                      6386  6769  7374
# 1                      2536  7578  5619
# 2                      3306  6766  8695
#
# training size: 165087 testing size: 55029
# For fold:  3 , the correlations is: 0.43138541292
# Predicted interaction     0     1      2
# Actual interaction
# 0                      5680  5383  13241
# 1                      2115  6910   5076
# 2                      2808  5773   8043

