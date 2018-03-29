#####  codes that maybe needed


#####  functions
# read genes and terms information
def reader():
    df = pd.read_csv('/home/wenyanli/cmsc828p/project1c-predicting-genetic-interactions/data/real_data/input/gene_term.csv')
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

def evaluate_randomly(eval_batch_size, data):
    input_batches, input_lengths, target_batches, target_lengths = random_batch(eval_batch_size, data=data)

    verb_predictor.train(False)

    # run through predictor model
    predictor_output, predictor_attn = verb_predictor(input_batches, input_lengths, None)

    loss = torch.nn.MSELoss()
    print('eval_loss', loss.data[0])

    if use_cuda:
        predicted = predictor_output.topk(1)[1].data.cpu().numpy().tolist()
        true = target_batches.transpose(0, 1).contiguous().data.cpu().numpy().tolist()
        attn = predictor_attn.squeeze(1).data.cpu().numpy().tolist()
    else:
        predicted = predictor_output.topk(1)[1].data.numpy().tolist()
        true = target_batches.transpose(0, 1).contiguous().data.numpy().tolist()
        attn = predictor_attn.squeeze(1).data.numpy().tolist()

    verb_predictor.train(True)

    return predicted, true, predictor_attn




###### in main():
# read data from GO terms
genes, terms, term_set = reader()
num_genes = len(genes)
print("number of genes:", len(genes))


