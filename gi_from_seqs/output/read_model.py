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
            labels.append('N')#()
        elif pred > 2:
            labels.append('P')#()
        else:
            labels.append('NI')#()
    return labels

def label_single(pred):
    if pred < -2.5:
        label = 'N'
    elif pred > 2:
        label = 'P'
    else:
        label = 'NI'
    return label

def get_correlation(truth, pred):
    assert len(truth) == len(pred)
    correlation = scipy.stats.pearsonr(truth,pred)
    return correlation

def convert2protein(index2protein, seq):
    proteins = [index2protein[s] for s in seq]
    return proteins

def find_most_attention(k,attn_seq,test_data):
    """find indexes in the seq that with most attention"""
    index_attn_seq = (nlargest(k, enumerate(attn_seq), key=lambda x: x[1]))
    sorted_seq = sorted(index_attn_seq,key=lambda x:x[0])
    indexes = [pair[0] for pair in sorted_seq]
    attns = [pair[1] for pair in sorted_seq]
    data_seq = [test_data[0][i] for i in indexes]
    return data_seq,attns,indexes



def plot_score():
    num_bins = 13
    num_in_each = int(len(pred)/num_bins)
    result = [(p,t) for (p,t) in zip(pred,true)]
    sorted_by_pred = sorted(result, key=lambda tup: tup[0])

    # get chunks
    chunks = [sorted_by_pred[x:x+num_in_each] for x in range(0, len(sorted_by_pred), num_in_each)]
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

def show_attention(input_sentence,predicted,true, attention, indexes):
    # plot attention for a single sentence
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.asarray([attention]),cmap ='bone')
    fig.colorbar(cax)

    # set up axes
    pair = [(i,p) for (i,p) in zip(indexes,input_sentence)]
    output_display = "".join(["("]+true+["): "]+predicted)
    ax.set_xticklabels(['']+pair,rotation =60)
    ax.set_yticklabels(['']+[output_display])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def plot_atten():
    for i in range(2):
        randi = random.randint(0,len(test_data))
        data_seq, attns,indexes = find_most_attention(10, attn[randi],test_data[randi])
        show_attention(convert2protein(index2protein,data_seq),[label_single(pred[randi])],
                       [label_single(true[randi])],attns,indexes)

def show_loss_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


# Get output data
OUTPUT_DIR = '/home/wenyanli/cmsc828p/project1c-predicting-genetic-interactions/gi_from_seqs/output/22/'
with open(os.path.join(OUTPUT_DIR,'output.pickle'),'rb') as f:
    pred,true,attn = pickle.load(f)

# Get training_loss
with open(os.path.join(OUTPUT_DIR,'training_loss.pickle'),'rb') as f:
    training_loss = pickle.load(f)

# Get protein_vocab
with open(os.path.join(OUTPUT_DIR,'protein_vocab.pickle'),'rb') as f:
    index2protein = pickle.load(f)
# print(index2protein)

# Get test data
with open(os.path.join(OUTPUT_DIR,'test_data.pickle'),'rb') as f:
    test_data = pickle.load(f)
# calculate correlation
print("correlation:",get_correlation(true,pred))

# calculate confusion matrix
true_label = label_interaction(true)
pred_label = label_interaction(pred)
confusion_matrix = pd.crosstab(pd.Series(true_label), pd.Series(pred_label), rownames=['Actual interaction'], colnames=['Predicted interaction'])
print(confusion_matrix,"\n")

# calculate precision and recall
from sklearn.metrics import classification_report
print(classification_report(true_label,pred_label))

index2protein[0] = '<PAD>'

# plotting
show_loss_plot(training_loss)
plot_score()
plot_atten()