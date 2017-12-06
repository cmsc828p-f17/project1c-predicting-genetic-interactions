import time
import math
import socket
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import scipy
import pickle
import random
import numpy as np
from sklearn.decomposition import IncrementalPCA


hostname = socket.gethostname()
use_cuda = torch.cuda.is_available()


floyd_flag = True
if floyd_flag:
    import process
else:
    import gi_from_dna.codes.process as process
# load equal length input data from file
train_data,dev_data,test_data = process.load_one_hot_data(floyd_flag)

def tensor2variable(tensor):
    result = Variable(torch.from_numpy(tensor).float())  # MAXLEN * 1 "column vector

    if use_cuda:
        return result.cuda()
    else:
        return result

def get_batch(batch_size, data, idx=0):
    # get a batch of data from index == idx
    gene1_seq = np.zeros([batch_size,4,1000])
    gene2_seq = np.zeros([batch_size,4,1000])
    target_scores = np.zeros(batch_size)

    # choose pair from index
    for i in range(batch_size):
        pair = data[idx]
        gene1_seq[i,:,:] = pair[0][0]
        gene2_seq[i,:,:] = pair[0][1]
        target_scores[i] = pair[1]
        idx += 1

    # Turn padded arrays into (batch x 4 x max_len) tensors,
    gene1_var = tensor2variable(gene1_seq)
    gene2_var = tensor2variable(gene2_seq)
    target_var = tensor2variable(target_scores)#.transpose(0)
    return gene1_var,gene2_var,target_var

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

class Conv(nn.Module):

    def __init__(self,num_kernels,stride):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv1d(4, num_kernels, stride)

    def forward(self, x1,x2):
        # Max pooling over a (2, 2) window
        x1 = F.max_pool1d(F.relu(self.conv1(x1)),stride=20,kernel_size=20)
        x2 = F.max_pool1d(F.relu(self.conv1(x2)),stride=20,kernel_size=20)
        # transpose x into L x B x Hidden
        x_out1 = x1.transpose(2,1).transpose(0,1)
        x_out2 = x2.transpose(2,1).transpose(0,1)
        # caoncatenate the two genes
        x_out = torch.cat((x_out1,x_out2),0)
        return x_out

# define encoder module
class EncoderRNN(nn.Module):
    def __init__(self,embed_size, stride,hidden_size, n_layers=2, dropout=0.2,is_bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.embed_size= embed_size
        self.stride = stride
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.is_bidirectional = is_bidirectional

        self.conv = Conv(embed_size,stride)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=is_bidirectional)

    def forward(self,x1,x2, hidden=None):
        x_out = self.conv(x1,x2) # L x Batch_size x embed_size
        outputs, hidden = self.gru(x_out, hidden)

        if self.is_bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        else:
            outputs = outputs[:, :, :self.hidden_size]

        return outputs,hidden

# build attention module
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn,self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
        if use_cuda:
            attn_energies = attn_energies.cuda()

        # for each batch of encoder outputs
        for b in range(this_batch_size):
            # calculate energy for each encoder output
            for i in range(max_len):
                # print("hidden[b,:]:",hidden[b,:],"encoder", encoder_outputs[i,b])
                attn_energies[b, i] = self.score(hidden[b,:], encoder_outputs[i,b])
                # hidden[b,:] (size=hidden_size)  ; encoder_outputs[i,b] (size = hidden_size)

        # Normalize energies to weights in range 0 to 1, resize to  1 x B X S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self,x, y):
        if self.method == 'dot':
            energy = torch.dot(x,y)
            return energy

        elif self.method == 'general':
            energy = self.attn(y)
            energy = x.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((x, y),1))
            energy = self.self.v.dot(energy)
            return energy

# dense layer predictor model
class Predictor(nn.Module):
    def __init__(self, method, hidden_size, output_size=1, n_layers=3):
        super(Predictor, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.method = method

        # Define layers
        self.attn = Attn(method, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden, encoder_outputs):
        # print("last_hidden[-1]",last_hidden[-1]) # 3x6
        attn_weights = self.attn(encoder_last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1)) # B X 1 X hidden_size
        context = context.transpose(0,1) # 1 x B x  hidden_size
        output = context.squeeze(0) # BxN
        output = self.out(output) # B x hidden_size
        return output, attn_weights #, context     ### uncomment when debugging


class IPmodel(nn.Module):
    def __init__(self, method, embed_size, stride,hidden_size, output_size=1,dropout = 0.1,
                 encoder_layers = 2, bidirectional=True):
        super(IPmodel, self).__init__()

        # Define parameters
        self.method = method
        self.embed_size = embed_size
        self.stride = stride
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.output_size = output_size

        # Define layers
        self.encoder = EncoderRNN(embed_size,stride,hidden_size,encoder_layers, dropout,bidirectional)
        self.out = Predictor(method,hidden_size)
    def forward(self,x1,x2, hidden=None):
        encoder_outputs, encoder_hidden = self.encoder(x1,x2)
        # Prepare predictor input
        predictor_input = encoder_hidden[:1]  # 1 x batch_size x hidden_size
        # predict
        predictor_output, predictor_attn_weights = self.out(predictor_input, encoder_outputs)
        return predictor_output,predictor_attn_weights

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def get_correlation(truth, pred):
    assert len(truth) == len(pred)
    correlation = scipy.stats.pearsonr(truth,pred)
    return correlation[0]   # return only the correlation


attnModel = 'dot'
embedSize = 200
stride  = 10
hiddenSize = 128
dropout = 0.2
batchSize = 64
outputSize = 1

print_every = 100

# gene1_var, gene2_var, target_var = get_batch(batchSize,train_data,idx=0)
# print(gene1_var.size())  Batch x 4 x 1000

# initialize model
interaction_predictor = IPmodel(attnModel,embedSize,stride,hiddenSize,outputSize,dropout=dropout)
interaction_predictor_optimizer = optim.Adam(interaction_predictor.parameters(),lr=0.00001)

# Move models to GPU
if use_cuda:
    interaction_predictor.cuda()

# keep track of time elapsed and running averages
start = time.time()
print_loss_total = 0 # reset every print_every
plot_loss_total = 0 # reset every plot_every


# defining a training iteration
def train(gene1_var, gene2_var, target_var,interaction_predictor, interaction_predictor_optimizer):
    # zero gradients of the optimizers
    interaction_predictor_optimizer.zero_grad()
    # run words through encoder
    predictor_output,predictor_attn_weights = interaction_predictor(gene1_var, gene2_var)
    # print("predicted:", predictor_output,"\n", "target:", target_var)
    # Loss calculation and back-propagation
    loss = torch.nn.MSELoss()
    output = loss(predictor_output,target_var)  # output, target
    output.backward()
    interaction_predictor_optimizer.step() # update parameters with optimizers

    return output.data[0]


def evaluate(batch_size, data):
    batch_i = 0
    num_of_records = len(data)
    num_batches = int(num_of_records // batch_size)

    predicted_all = []
    true_all = []
    attn_all = []
    # loss_all = []

    while batch_i <  num_batches:
        start_i = batch_i * batch_size

        gene1_var, gene2_var, target_var = get_batch(batch_size, data,idx=start_i)

        interaction_predictor.train(False)

        # run through prediction model
        predictor_output, predictor_attn = interaction_predictor(gene1_var,gene2_var)
        # print("predictor_output:", predictor_output.data.view(1,-1).tolist()[0])

        loss = torch.nn.MSELoss()
        output = loss(predictor_output, target_var)
        print('eval_loss', output.data[0])

        if use_cuda:
            predicted = predictor_output.data.cpu().view(1,-1).tolist()[0]
            true = target_var.contiguous().data.cpu().numpy().tolist()
            attn = predictor_attn.squeeze(1).data.cpu().numpy().tolist()
        else:
            predicted = predictor_output.data.view(1,-1).tolist()[0]
            true = target_var.contiguous().data.numpy().tolist()
            # print(len(predicted),len(true))
            attn = predictor_attn.squeeze(1).data.numpy().tolist()

        predicted_all.extend(predicted)
        true_all.extend(true)
        attn_all.extend(attn)
        # loss_all.extend(output.data[0])
        batch_i += 1

    interaction_predictor.train(True)
    return predicted_all, true_all,attn_all#, loss_all

def train_minibatch(train_data,eval_data,batch_size,epoch,start_epoch = 0, print_loss_total=0, plot_loss_total=0):
    batch_i = 0  #initialize batch index
    num_of_records = len(train_data)
    num_batches = int(num_of_records//batch_size)
    evaluate_every = num_batches
    loss_rec = []
    acc_all = []
    # shuffle the data at each epoch
    random.shuffle(train_data)

    while batch_i < num_batches:
        print("batch:", batch_i, "total_batches:", num_batches, int(batch_i * 100 // num_batches), "%")
        start_i = batch_i * batch_size
        gene1_var, gene2_var, target_var = get_batch(batch_size,train_data, idx=start_i) # get training data for this cycle
        # print(target_var)
        # # run the train function
        loss = train(gene1_var, gene2_var, target_var,interaction_predictor,interaction_predictor_optimizer)

        # keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if (batch_i+1) % print_every == 0:
            print_loss_avg = print_loss_total/print_every
            loss_rec.append(print_loss_avg)
            print_loss_total = 0
            print_summary ='Time:%s, Batch:(%d %d%%), Avg_loss:%.4f' %\
                           (time_since(start, batch_i / num_batches),
                            batch_i,batch_i / num_batches * 100, print_loss_avg)
            print(print_summary)

        if (batch_i+1) % evaluate_every == 0:   # evaluate on dev set after each epoch
            predicted, true, attn = evaluate(batch_size, eval_data)
            # print("predicted:", predicted, len(predicted),"\n","true:",true,len(true))
            acc = get_correlation(true,predicted)
            print("accuracy: ", acc)
            acc_all.append(acc)  # save all the accuracies
            best_accuracy = max(acc_all)
            print("accuracy: ", acc)

            # Get bool not ByteTensor
            is_best = bool(acc >= best_accuracy)
            # Get greater Tensor to keep track best acc
            best_accuracy = max(acc, best_accuracy)
            # Save checkpoint if is a new best
            save_checkpoint({
                'epoch': start_epoch + epoch + 1,
                'state_dict': interaction_predictor.state_dict(),
                'best_accuracy': best_accuracy
            }, is_best)

        batch_i += 1  # update batch index
    return loss_rec,acc_all

num_epochs = 15
losses_all = []
eval_acc_all = []
for epoch in range(num_epochs):
    print("Epoch: %d/%d"%(epoch,num_epochs))
    loss_rec, acc_all= train_minibatch(train_data,dev_data,batchSize,epoch)
    losses_all.extend(loss_rec)
    eval_acc_all.extend(acc_all)

predicted, true, attn= evaluate(batchSize, test_data)
acc = get_correlation(true,predicted)
print("test accuracy: ", acc)


## save model and output for further analysis
with open('/output/output.pickle', 'wb') as f:
    pickle.dump([predicted, true, attn], f)
with open('/output/test_data.pickle', 'wb') as d:
    pickle.dump(test_data, d)
with open('/output/training_loss.pickle','wb') as f:
    pickle.dump(losses_all,f)
with open('/output/eval_accuracy.pickle','wb') as f:
    pickle.dump(eval_acc_all)

torch.save(interaction_predictor.state_dict(), '/output/vp.dat')

