import pickle
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.ticker as ticker
from heapq import nlargest

def label_interaction(preds):
    labels = []
    for pred in preds:
        if pred < -2.5:
            labels.append('N')
        elif pred > 2:
            labels.append('P')
        else:
            labels.append('NI')
    return labels
def plot_score():
    num_bins = 10
    num_in_each = int(len(pred)/num_bins)
    result = [(p,t) for (p,t) in zip(pred,true)]
    sorted_by_pred = sorted(result, key=lambda tup: tup[0])

# get chunks
    chunks = [sorted_by_pred[x:x+num_in_each] for x in range(0, len(sorted_by_pred),
    num_in_each)]
    preds = np.zeros([num_in_each,num_bins])
    trues = np.zeros([num_in_each,num_bins])
    for i in range(num_bins):
        for j in range (num_in_each):
            preds[j][i] = chunks[i][j][0]
            trues[j][i] = chunks[i][j][1]
            mean = np.mean(preds, axis=0)
            cols = ['{0:.2f}'.format(m) for m in mean]

# plotting
    df = pd.DataFrame(trues,columns=cols)
    df.plot.box()
    plt.xlabel('Predicted Interaction Score')
    plt.ylabel('Measured Interaction Score')
    plt.title('Scores')
    plt.show()

# Get output data
with open(('output.pickle'),'rb') as f:
    pred,true,attn = pickle.load(f)

# Get training_loss
with open(('training_loss.pickle'),'rb') as f:
    training_loss = pickle.load(f)

# Get protein_vocab
with open(('protein_vocab.pickle'),'rb') as f:
    index2protein = pickle.load(f)
# print(index2protein)
true_label = label_interaction(true)
pred_label = label_interaction(pred)

from sklearn.metrics import classification_report
print(classification_report(true_label, pred_label))

# plot_score()

