import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import scipy
from sklearn.model_selection import KFold
import os

# Set random seed
np.random.seed(0)

# read interactions
def read_interactions():
    term_set = []
    with open('./example-hierarchy-sets.tsv') as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      for row in reader:
        term_set.append(row)
    return term_set

# read genes
def read_genes():
    genes = []
    with open('./example-gene-names.txt') as txtfile:
      reader = csv.reader(txtfile, delimiter='\n')
      for row in reader:
        genes.extend(row)
    return genes


def pair_genes(genes, num_genes, gene_interactions):
    # produce gene interaction pairs
    gene_pairs = []
    pair_scores = []
    # class_dict = {"positive" : 1, "negative":2,"non-interaction":0}

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            gene_pairs.append((genes[i], genes[j]))
            pair_scores.append(gene_interactions[i][j])
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

def construct_feature_vector(gene_pairs, gene_dict, pair_scores):
    feature_vector = np.zeros([4950, 100])  # 4950 x (99+1)  4950 gene combination(assume (g1,g2) is equivalent to (g2,g1))
    row_idx = 0
    for idx,(gene1,gene2) in enumerate(gene_pairs):
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
        print("For fold: ", k, ", the correlations is:", correlation[0])

        # label the interactions from the scores
        preds_label = label_interaction(pred_score)
        true_label = label_interaction(true_score)
        # generate confusion matrix
        confusion_matrix = pd.crosstab(pd.Series(true_label), pd.Series(preds_label), rownames=['Actual interaction'], colnames=['Predicted interaction'])
        print(confusion_matrix,"\n")


if __name__ == "__main__":
    # read terms
    term_set = read_interactions()

    # read gene list
    genes = read_genes()
    num_terms = len(term_set)  # 99 terms
    num_genes = len(genes)     # 100 genes

    # read interaction score matrix
    gene_interactions = np.load("./example-genetic-interactions.npy")

    # generate gene pairs and label the interactions
    gene_pairs, pair_scores = pair_genes(genes, num_genes, gene_interactions)

    # help to construct feature vector
    gene_dict = build_geneterm_dict(genes, term_set)

    # construct feature vector
    feature_vector = construct_feature_vector(gene_pairs,gene_dict, pair_scores)
    feature_vector = np.asarray(feature_vector)

    # separate into features and scores
    X = feature_vector[:,:-1]
    y = feature_vector[:,-1]

    # four-fold cross validation
    CrossValidation(X,y,4)

