import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ggplot import *
import scipy

# load 4 four validation result and calculate final result
pred_score = []
true_score = []
correlations =  []
for i in range(4):
    p,t,c = pickle.load(open("./final_result/{0}th_result.pickle".format(i), "rb"))
    pred_score.extend(p)
    true_score.extend(t)
    correlations.append(c[0])
# print(correlations)
# print(len(pred_score),len(true_score))

final_correlation = scipy.stats.pearsonr(true_score, pred_score)
print("Final correlations is:", final_correlation[0])


# plotting
num_bins = 13
num_in_each = int(len(pred_score)/num_bins)
result = [(p,t) for (p,t) in zip(pred_score,true_score)]
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
plt.savefig("./final_results/scores.png")
