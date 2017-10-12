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

def read_sets():
    term_set = []
    with open('./input/term_sets.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter='\n')
      for row in reader:
        term_set.append(row)
    return term_set
# read genes
def read_genes():
    df = pd.read_csv('./input/gene_term.csv')
    genes = df['Gene'].tolist()
    #terms = df['Term'].tolist()
    return genes

def obtain_interactions():
    interactions = []
    with open('./collins-sc-emap-gis.tsv') as tsvfile:
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

def pair_genes(genes,interactions_matrix,interaction_genes):
    # produce gene interaction pairs
    gene_pairs = []
    pair_scores = []
    # class_dict = {"positive" : 1, "negative":2,"non-interaction":0}
    num_genes = len(genes)

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gene_pairs.append((genes[i], genes[j]))
            if genes[i] or genes[j] not in interaction_genes:
                pair_scores.append(0)
            else:
                pair_scores.append(interactions_matrix[i][j])
    return gene_pairs, pair_scores

def build_geneterm_dict(genes, term_set):
    # build gene dict which returns the terms that specific gene belongs
    gene_dict = {}
    for gene in genes:
        for idx, term in enumerate(term_set):
            if gene in term:
                if gene not in gene_dict:
                    gene_dict[gene] = [idx]
                else:
                    gene_dict[gene].append(idx)
    return gene_dict

def construct_feature_vector(gene_pairs, gene_dict, pair_scores, num_terms,genes_in_terms):
    feature_vector = np.zeros([len(gene_pairs), num_terms+1])
    # only interaction pairs(664x663/2) x (5125+1)  4950 gene combination(assume (g1,g2) is equivalent to (g2,g1))
    row_idx = 0
    for idx,(gene1,gene2) in enumerate(gene_pairs):
        if gene1 in genes_in_terms and gene2 in genes_in_terms:
            terms_i = gene_dict[gene1]
            terms_j = gene_dict[gene2]
            feature_vector[row_idx][terms_i] += 1
            feature_vector[row_idx][terms_j] += 1
            feature_vector[row_idx][-1] = pair_scores[idx]
            row_idx +=1
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
        print("For fold: ", k, ", the correlations is:", correlation[0],r2_score(true_score,pred_score))

        # label the interactions from the scores
        preds_label = label_interaction(pred_score)
        true_label = label_interaction(true_score)
        # generate confusion matrix
        confusion_matrix = pd.crosstab(pd.Series(true_label), pd.Series(preds_label), rownames=['Actual interaction'], colnames=['Predicted interaction'])
        print(confusion_matrix,"\n")


if __name__ == "__main__":

    # read genes in terms
    term_set = read_sets()
    num_terms = len(term_set)     # 5125
    print(term_set[1])

    # # read genes
    # genes = read_genes()
    # num_genes = len(genes)    # 320877
    # unique_genes = list(set(genes))
    # print("total number of genes: ", len(genes),"num unique genes:", len(unique_genes))
    #
    # # get genes that belong to some terms
    # tmp = []
    # tmp.extend(terms for terms in term_set)
    # genes_in_terms = list(set(tmp))   # 4682 (< num of unique genes)
    #
    # # obtain interactions
    # interaction_genes, interactions_matrix = obtain_interactions()
    #
    # # help to construct feature vector
    # gene_dict = build_geneterm_dict(genes_in_terms, term_set)
    #
    # gene_pairs, pair_scores = pair_genes(genes_in_terms, interactions_matrix, interaction_genes)
    #
    # # construct feature vector only for gene pairs with genetic interaction
    # feature_vector = construct_feature_vector(gene_pairs,gene_dict, pair_scores, num_terms,genes_in_terms)
    # feature_vector = np.asarray(feature_vector)   # (4682*4681/2)x5126
    # # print(feature_vector.shape)
    #
    # # separate into features and scores
    # X = feature_vector[:,:-1]
    # y = feature_vector[:,-1]
    #
    # #four-fold cross validation
    # CrossValidation(X,y,4)

