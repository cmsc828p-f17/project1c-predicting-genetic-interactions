import os
import cPickle as pickle
import numpy
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from scipy.stats import pearsonr
import random

# Flag to control execution for simulated data or the real data
SIMULATED_DATA = True

# Featurize a given gene1, gene2, score pair, given
# the gos dictionary and ordered list of gos (keys)
def featurize(keys, gos, gene1, gene2, num):
    feature_vector = [0] * num
    for idx in xrange(0, num):
        tot = 0
        go = keys[idx]
        tot += 1 if gene1 in gos[go] else 0
        tot += 1 if gene2 in gos[go] else 0
        feature_vector[idx] = tot
    return feature_vector

# Read in gos data from saved pickle file if exists, otherwise parse it and save pickle
if not SIMULATED_DATA:
    if not os.path.exists("gos.pickle"):
        print("Beginning go parsing.")
        gos = {}
        with open("mmc.csv") as f:
            for line in f:
                gene, go = line.strip().split(",")
                if go not in gos:
                    gos[go] = set()
                gos[go].add(gene)
            print("Saving gos.")
            pickle.dump(gos, open("gos.pickle", "wb"))
    else:
        print("Reading computed gos from file.")
        gos = pickle.load(open("gos.pickle", "rb"))

    keys = gos.keys()
    num = len(keys)
    ys = []
    xs = []
    idx = 0
    
    if not os.path.exists("inputs.pickle"):
        print("Beginning gene featurization for %d gos." % num)
        with open("collins.tsv") as f:
            for line in f:
                print("%d\r"%idx),
                idx += 1
                gene1, gene2, score, _ = line.split()
                ys.append(float(score))
                xs.append(featurize(keys, gos, gene1, gene2, num))
            print("Saving gene featurization.")
            pickle.dump((xs, ys), open("inputs.pickle", "wb"))
    else:
        print("Reading gene featurization from file.")
        xs, ys = pickle.load(open("inputs.pickle", "rb"))
else:
    if not os.path.exists("simulated_gos.pickle"):
        print("Beginning simulated go parsing.")
        gos = {}
        with open("data/examples/example-hierarchy-sets.tsv") as f:
            go = 0
            for line in f:
                go += 1
                genes = line.strip().split()
                gos[go] = genes
            print("Saving gos.")
            pickle.dump(gos, open("simulated_gos.pickle", "wb"))
    else:
        print("Reading simulated computed gos from file.")
        gos = pickle.load(open("simulated_gos.pickle", "rb"))

    keys = gos.keys()
    num = len(keys)
    ys = []
    xs = []
    idx = 0
    if True:
#    if not os.path.exists("simulated.pickle"):
        with open("data/examples/example-gene-names.txt") as f:
            idx = 0
            lookup = {}
            # Collect all of the gene names into a lookup list to associate gene name with index
            for line in f:
                lookup[line.strip()] = idx
                idx += 1
        # Read in the gene scores
        gene_scores = numpy.load("data/examples/example-genetic-interactions.npy")
        genes = lookup.keys()
        # Iterate over every pair of genes, get it's score from gene_scores, and collect that info for training.
        for i in range(100):
            for j in range(i+1, 100):
                score = gene_scores[i][j]
                ys.append(float(score))
                idx = 0
                gene_v = [0]*100
                for gene_set in genes:
                    gene_v[idx] += 1 if ("Gene-%d" % (i + 1)) in gene_set else 0
                    gene_v[idx] += 1 if ("Gene-%d" % (j + 1)) in gene_set else 0
                    idx += 1
                xs.append(gene_v)
            #print gene1, gene2, score
        pickle.dump((xs, ys), open("simulated.pickle", "wb"))
#    else:
#        print("Reading simulated gene featurization from file.")
#        xs, ys = pickle.load(open("simulated.pickle", "rb"))
shuffler = list(zip(xs, ys))
random.shuffle(shuffler)
xs, ys = zip(*shuffler)


print("Have gos, xs, ys. ")
print("Creating regressor with 10 trees, and cross validating 4 times.")
regr = RandomForestRegressor(n_estimators=10)
preds = cross_val_predict(regr, xs, ys, cv=4)
print pearsonr(preds, ys)[0]
