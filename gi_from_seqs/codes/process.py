import pandas as pd
import os
import csv
import random
import pickle
from collections import defaultdict
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

floyd_flag = True

def sequence_reader(INPUT_DIR, FILE):
    genes = []
    line_idxs = []
    text = ''
    gene_protein_dict = {}

    input_file = open(os.path.join(INPUT_DIR,FILE), "r")
    csvReader = csv.reader(input_file)
    for line_i, row in enumerate(csvReader):
        if '>' in row[0]:
            gene_name = row[0][1:]
            genes.append(gene_name)
            line_idxs.append(line_i)
            text = text + ","
        else:
            text = text + row[0]

    protein_seqs = text.split(",")[1:]

    for i, gene in enumerate(genes):
        gene_protein_dict[gene] = protein_seqs[i]
    return gene_protein_dict, protein_seqs,genes

def dna_sequence_reader(INPUT_DIR, FILE):
    genes = []
    line_idxs = []
    text = ''
    gene_seq_dict = {}

    input_file = open(os.path.join(INPUT_DIR,FILE),"r")
    csvReader = csv.reader(input_file)
    for line_i, row in enumerate(csvReader):
        if '>' in row[0]:
            gene_name = row[0].split(" ")[0][17:]
            genes.append(gene_name)
            line_idxs.append(line_i)
            text = text + ","
        else:
            text = text + row[0]

    dna_seqs = text.split(",")[1:]

    for i, gene in enumerate(genes):
        gene_seq_dict[gene] = dna_seqs[i]
    return gene_seq_dict, dna_seqs, genes



def obtain_interactions(floyd):
    interactions = []
    if floyd:
        FILE = '/input/collins-sc-emap-gis.tsv'
    else:
        FILE = '/home/wenyanli/cmsc828p/project1c-predicting-genetic-interactions/data/real_data/input/collins-sc-emap-gis.tsv'
    with open(FILE) as tsvfile:
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

def get_fixed_length_sequence(interaction_genes,gene_protein_dict,seq_len):
    filtered_genes = {}
    for gene in interaction_genes:
        if gene in gene_protein_dict.keys():
            filtered_genes[gene] = gene_protein_dict[gene]

    # deal with protein sequence length
    fixed_len_protein = {}
    for gene in filtered_genes:
        if len(filtered_genes[gene]) > seq_len:
            fixed_len_protein[gene] = filtered_genes[gene][:seq_len]
        else:
            fixed_len_protein[gene] = filtered_genes[gene]
    return fixed_len_protein

def pad_seq(seq, max_length):
    PAD_token = 0
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def construct_feature_vector(gene_pairs, gene_dict, pair_scores, protein2index, seq_len):
    feature_vector = []# simply concatenate the two vectors
    target_score = []
    # only interaction pairs (num_interactions) x (5125+1)  assume (g1,g2) is equivalent to (g2,g1)
    row_idx = 0
    for idx,(gene1,gene2) in enumerate(gene_pairs):
        if gene1 in gene_dict and gene2 in gene_dict:
            protein_i = gene_dict[gene1]
            protein_j = gene_dict[gene2]
            f = pad_seq([protein2index[p] for p in protein_i],seq_len) + pad_seq([protein2index[p] for p in protein_j],seq_len)
            feature_vector.append(f)
            target_score.append(float(pair_scores[idx]))
            row_idx +=1
    return feature_vector, target_score

def build_protein_vocab(fixed_len_protein_dict,floyd_flag=False):
    proteins = ''
    for p in fixed_len_protein_dict.values():
        proteins += p
    unique_proteins = set(proteins)
    protein2index = {}
    index2protein = {}
    for idx,p in enumerate(unique_proteins):
        protein2index[p] = int(idx)+1
    for k,v in protein2index.items():
        index2protein[v] = k

    # save index2protein
    if floyd_flag:
        with open('/output/protein_vocab.pickle', 'wb') as f:
            pickle.dump(index2protein, f)
    return protein2index, index2protein

def pairing_data(X, Y):
    paired_data = [(x,y) for (x, y) in zip(X, Y)]
    return paired_data

def label_interaction(preds):
    labels = []
    for pred in preds:
        if pred < -2.5:
            labels.append('negative')#()
        elif pred > 2:
            labels.append('positive')#()
        else:
            labels.append('no-interaction')#()
    return labels

def select_indexes(scores):
    negative_indexes = []
    positive_indexes = []
    non_inter_indexes = []
    for i, score in enumerate(scores):
        if score < -2.5:
            negative_indexes.append(i)
        elif score > 2:
            positive_indexes.append(i)
        else:
            non_inter_indexes.append(i)

    # print(len(negative_indexes),len(positive_indexes),len(non_inter_indexes))
    return positive_indexes,negative_indexes,non_inter_indexes

def get_inputs(feature_vectors, target_scores):
    pos_indexes, neg_indexes, non_indexes = select_indexes(target_scores)
    input_non_indexes = random.sample(non_indexes, (len(non_indexes) // 5))
    all_indexes = pos_indexes+neg_indexes+input_non_indexes

    input_vectors = [feature_vectors[i] for i in all_indexes]
    targets = [target_scores[i] for i in all_indexes]
    return input_vectors,targets



def load_data(floyd_flag=False):
    if floyd_flag:
        INPUT_DIR = "/input/"
    else:
        INPUT_DIR = "/home/wenyanli/cmsc828p/project1c-predicting-genetic-interactions/gi_from_seqs/input/"

    FILE = "protein_seqs"
    # obtain interactions information
    interaction_genes, interactions_matrix,gene_pairs,pair_scores = obtain_interactions(floyd_flag)
    # print(len(gene_pairs),len(pair_scores))

    gene_protein_dict, protein_seqs, genes = sequence_reader(INPUT_DIR,FILE)

    SEQ_LEN = 150
    # obtain fixed protein dict
    fixed_length_protein_dict = get_fixed_length_sequence(interaction_genes, gene_protein_dict,SEQ_LEN)
    # print([len(s) for s in list(fixed_length_protein_dict.values())[0:10]])
    # build protein-index vocab
    protein2index, index2protein = build_protein_vocab(fixed_length_protein_dict,floyd_flag)

    # transfer amino-acid seq to index seq (cancatenate the two vectors)
    feature_vectors, target_scores= construct_feature_vector(gene_pairs, fixed_length_protein_dict, pair_scores,protein2index,SEQ_LEN)
    input_vectors,targets = get_inputs(feature_vectors,target_scores)
    # print(len(feature_vectors[0]), len(input_vectors))
    # split into train, test, dev
    train_X, test_X,train_Y,test_Y = train_test_split(input_vectors,targets,test_size=0.2)
    train_X, dev_X, train_Y, dev_Y= train_test_split(train_X,train_Y, test_size=0.1)
    # print("number of records in training: %d, dev: %d, test: %d"%(len(train), len(dev), len(test)))
    train = pairing_data(train_X,train_Y)
    dev = pairing_data(dev_X,dev_Y)
    test = pairing_data(test_X,test_Y)

    input_size = len(protein2index)
    return train, dev, test, input_size

def load_dna_data(floyd_flag=False):
    if floyd_flag:
        INPUT_DIR = "/input/"
    else:
        INPUT_DIR = "/home/wenyanli/cmsc828p/project1c-predicting-genetic-interactions/gi_from_seqs/input"

    FILE = "dna_seqs"
    gene_seq_dict, dna_seqs, genes = dna_sequence_reader(INPUT_DIR, FILE)
    interaction_genes, interactions_matrix, gene_pairs, pair_scores = obtain_interactions(floyd_flag)
    SEQ_LEN = 500
    # obtain fixed length seq dict
    fixed_length_dict = get_fixed_length_sequence(interaction_genes, gene_seq_dict, SEQ_LEN)
    base2index, index2base = build_protein_vocab(fixed_length_dict, floyd_flag)


