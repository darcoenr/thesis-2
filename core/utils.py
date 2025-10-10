import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from collections import defaultdict
from typing import Callable

import core.models as models

class AbDataset(Dataset):
    def __init__(self, df):
        self.ids = df['pair_id'].to_numpy()
        self.sequences = df[['heavy', 'light']].apply(lambda x: ' '.join(x['heavy']) + ' [SEP] ' + ' '.join(x['light']), axis=1).to_numpy()
        self.labels = df['class'].to_numpy()
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.ids[idx], self.sequences[idx], self.labels[idx]

class AbTestDataset(Dataset):
    def __init__(self, df):
        self.ids = df['pair_id'].to_numpy()
        
        self.heavy_id = df['heavy_id'].to_numpy()
        self.heavy = df['heavy'].apply(lambda x: ' '.join(x)).to_numpy()

        self.light_id_pos = df['light_id_pos'].to_numpy()
        self.light_pos = df['light_pos'].apply(lambda x: ' '.join(x)).to_numpy()

        self.light_id_neg = df['light_id_neg'].to_numpy()
        self.light_neg = df['light_neg'].apply(lambda x: ' '.join(x)).to_numpy()

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return (self.ids[idx], self.heavy_id[idx], self.light_id_pos[idx],  self.light_id_neg[idx], 
                self.heavy[idx] + ' [SEP] ' + self.light_pos[idx],
                self.heavy[idx] + ' [SEP] ' + self.light_neg[idx])
    
class TablePrinter:
    def __init__(self, metrics_names, number_of_significant_digits):
        self.count = 0
        self.metrics_names = metrics_names
        self.number_of_significant_digits = number_of_significant_digits
        self.spaces_for_each_metric = 9 + number_of_significant_digits*2
        self.space_for_each_fold = len(metrics_names)*self.spaces_for_each_metric + (len(metrics_names) - 1) # This term is for accounting the |
    def print_header(self):
        # First row
        print(f"{'It.':^{6}}", end='')
        print(f"|{'TRAIN':^{self.space_for_each_fold}}|", end='')
        print(f"{'EVAL':^{self.space_for_each_fold}}|")
        # Second row
        print(f"{'-'*6}", end='')
        print(f"|{'-'*self.space_for_each_fold}|{'-'*self.space_for_each_fold}|")
        # Third row
        print(f"{' '*6}|", end='')
        for _ in range(2): # Repeat for each fold (train, eval)
            for metric in self.metrics_names:
                print(f"{metric:^{self.spaces_for_each_metric}}|", end='')
        print()
        # Fourth row
        print(f"{'-'*6}|", end='')
        for _ in range(2): # Repeat for each fold (train, eval)
            for metric in self.metrics_names:
                print(f"{'-'*self.spaces_for_each_metric}|", end='')
        print()
    def print_line(self, results_train, results_eval):
        self.count += 1
        print(f"{self.count:5} ", end='')
        print('|', end='')
        for d in [results_train,  results_eval]:
            for m in self.metrics_names:
                s = f"{d[m+'_mean'][-1]:.{self.number_of_significant_digits}f}"
                s += u" \u00B1 " 
                s += f"{d[m+'_std'][-1]:.{self.number_of_significant_digits}f}"
                print(f"{s:^{self.spaces_for_each_metric}}|", end='')
        print()

def get_dataset(location, *, subsample=None, frac=None, shuffle=False, seed=0):
    """Retrieve the dataset or a subsample of it."""

    dataset_hyperpar = {
        'name': location,
    }

    if subsample is not None and frac is not None:
        raise ValueError('Define either subsaple or frac but noth both.')

    print('Reading {}...'.format(location))
    df = pd.read_csv(location, index_col=0)
    print('Dataset of size {}'.format(len(df)))

    dataset_hyperpar['size'] = len(df)

    if frac is not None:
        subsample = int(len(df) * frac)

    if subsample is not None:
        df = df.sample(subsample, random_state=seed)
        print('Sampled a subset of size {}'.format(len(df)))

    if shuffle:
        df = df.sample(len(df))
    
    dataset_hyperpar['shuffled'] = shuffle
    dataset_hyperpar['frac'] = frac
    dataset_hyperpar['subsample'] = subsample
    dataset_hyperpar['seed'] = seed

    abd = AbDataset(df)
    return abd, dataset_hyperpar

def get_test_dataset(location, *, subsample=None, frac=None):
    """Retrieve the dataset or a subsample of it."""

    dataset_hyperpar = {
        'name': location,
    }

    if subsample is not None and frac is not None:
        raise ValueError('Define either subsaple or frac but noth both.')

    print('Reading {}...'.format(location))
    df = pd.read_csv(location, index_col=0)
    print('Dataset of size {}'.format(len(df)))

    dataset_hyperpar['size'] = len(df)

    if frac is not None:
        subsample = int(len(df) * frac)

    if subsample is not None:
        df = df.sample(subsample)
        print('Sampled a subset of size {}'.format(len(df)))

    dataset_hyperpar['frac'] = frac
    dataset_hyperpar['subsample'] = subsample

    abd = AbTestDataset(df)
    return abd, dataset_hyperpar

def retrieve_tokenizer(checkpoint):
    """Retrieve the tokenizer from Huggingface."""

    print('Tokenizer checkpoint: {}'.format(checkpoint))
    tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False)
    return tokenizer

def retrieve_model(model_name, *, checkpoint, number_of_layers, layer_size):
    if model_name == 'average':
        model = models.ClassificationFromAveraging(checkpoint=checkpoint, number_of_layers=number_of_layers, layer_size=layer_size)
        model_hyperpar = {'name': model_name, 
                          'checkpoint': checkpoint,
                          'number_of_layers': number_of_layers, 'layer_size': layer_size}
        return model, model_hyperpar
    if model_name == 'pooling':
        model = models.ClassificationFromPooling(checkpoint=checkpoint)
        model_hyperpar = {'name': model_name, 'checkpoint': checkpoint}
        return model, model_hyperpar
    
def retrieve_optimizer(optimizer_name, *, parameters, lr, momentum, weight_decay, beta1, beta2, epsilon):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(parameters, 
                              lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer_hyperpar = {
            'name': optimizer_name,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay
        }
        return optimizer, optimizer_hyperpar
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(parameters,
                               lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=epsilon)

        print(beta1, beta2)
        
        optimizer_hyperpar = {
            'name': 'adam',
            'lr': lr,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon
        }
        return optimizer, optimizer_hyperpar

def retrieve_lr_scheduler(lr_scheduler_name, *, optimizer, 
                          factor=None, patience=None, min_lr=None,
                          gamma=None, last_epoch=-1,
                          start_factor=1, end_factor=0.1, total_iters=100):
    if lr_scheduler_name == 'plateau':
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='max', factor=factor,
            patience=patience, threshold=1e-3, threshold_mode='abs', min_lr=min_lr,
            verbose=True
        )
        sched_hyperpar = {
            'name': lr_scheduler_name,
            'factor': factor,
            'patience': patience,
            'min_lr': min_lr
        }
        return sched, sched_hyperpar
    if lr_scheduler_name == 'exponential':
        sched = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=gamma, last_epoch=last_epoch
        )
        sched_hyperpar = {
            'name': lr_scheduler_name,
            'gamma': gamma,
            'last_epoch': last_epoch
        }
        return sched, sched_hyperpar
    if lr_scheduler_name == 'linear':
        sched = optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters
        )
        sched_hyperpar = {
            'name': lr_scheduler_name,
            'start_factor': start_factor,
            'end_factor': end_factor,
            'total_iters': total_iters
        }
        return sched, sched_hyperpar

def load_model(model_name, *, checkpoint, state_dict):
    if model_name == 'average':
        number_of_layers = len(state_dict) // 2
        layer_size = list(state_dict.items())[0][1].shape[0]
        
        model = models.ClassificationFromAveraging(checkpoint=checkpoint, number_of_layers=number_of_layers, layer_size=layer_size)
        model.load_state_dict(state_dict, strict=False)
        return model

def get_state_dict(name, model):
    if name == 'average':
        state_dict = model.state_dict()
        state_dict = {
            key: value for key, value in state_dict.items() if 'classification_head' in key
        }
        return state_dict

def tokenize_inputs(inputs, tokenizer, *, device):
    """Tokenize the inputs"""

    # Tokenize, extract relevant data and move to the device.
    tokens = tokenizer(inputs, add_special_tokens=True, padding=True, 
                       return_tensors='pt', return_special_tokens_mask=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    special_tokens_mask = tokens['special_tokens_mask'].to(device)

    return_dict = {'input_ids': input_ids, 'attention_mask': attention_mask,
                   'special_tokens_mask': special_tokens_mask}

    return return_dict

def compute_avg_embeddings(model, tokenizer, data_dl):
    """This function computes an embedded representation of the input sequneuces by averaging."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    
    results = []

    model = model.to(device)
    averager = models.AverageEmbedding().to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_dl, total=len(data_dl)):
            _, data, _ = data
            tokens = tokenizer(data, add_special_tokens=True, padding=True,
                               return_special_tokens_mask=True, return_tensors='pt')

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            special_tokens_mask = tokens['special_tokens_mask'].to(device)

            output = model(input_ids, attention_mask=attention_mask)
            embeddings = output['last_hidden_state']
            embeddings = averager(embeddings, special_tokens_mask)
            results.append(embeddings)
    embeddings = torch.cat(results).detach().to('cpu')
    return embeddings

def compute_metrics(target, prediction, metrics):
    """Compute the metrics given the prediction and the target."""

    return_dict = {}
    for key, value in metrics.items():
        return_dict[key] = value(target, prediction)
    return return_dict

def infer_batch(model, criterion, optimizer, batch, 
                *, 
                tokenizer, metrics, device, return_classification_results=False):
    """
    Infer the results of model on a single batch.
      
    If the model is in training mode, then compute the loss,
    perform backpropagation and perform ad change the parameters.
    Otherwise just compute the metrics.
    """

    pair_ids, inputs, labels = batch[0], batch[1], batch[2]
    
    # Tokenize the inputs
    input_dict = tokenize_inputs(inputs, tokenizer, device=device)

    # Compute model output
    logits = model(**input_dict)

    # Compute the loss
    loss = criterion(logits, labels.to(device))

    # Backpropagation
    if model.training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get the predicted labels and compute the metrics
    pred_labels = logits.detach().cpu().argmax(dim=1)
    metrics_dict = compute_metrics(labels, pred_labels, metrics)
    metrics_dict['loss'] = loss.item()

    if return_classification_results:
        return metrics_dict, pair_ids, pred_labels, labels
    else:
        return metrics_dict

def train(model, criterion, optimizer, sched,
          train_dl, eval_dl, 
          *, 
          n_epochs=1, 
          train_batches=None, eval_batches=None, evaluate_every=None, mean_lasts=10,
          metrics={},
          verbose=0,
          **kwargs):
    
    """
    Train the model for a specific number of epoch or batches.

    If train_batches, eval_batches and evaluate_every are all None
    then the model is trained on the whole training dataset for n_epochs
    times and evaluated on the whole validation dataset at the end
    of each epoch.

    If train_batches, eval_batches and evaluate_every are all different
    from None and n_epochs is None, then the model is trained for a total fo
    train_batches batches. Additionaly every evaluate_every batches the model
    is evaluate with eval_batches batches from the validation dataset.

    If n_epochs train_batches, eval_batches and evaluate_every are all specified
    then the training proceeds as the latter approach with the additional check that
    no more that n_epochs total pass on the training dataset are performed.
    """

    conditions = [train_batches  is None, 
                  eval_batches   is None, 
                  evaluate_every is None]
    
    # any(conditions)     <=> At least one is None
    # not any(conditions) <=> All are not None
    # all(conditions)     <=> All are None
    # not all(conditions) <=> At least one is not None 

    if n_epochs is None and conditions[0] is True:
        # Error! specify at least the number of epochs or the number of training batches
        raise ValueError('Either n_epoch is not None or all train_batches, eval_batches and evaluate_every are not None')
    elif n_epochs is not None and conditions[0] is True:
        # Just the number of epochs is specified
        train_batches = len(train_dl)*n_epochs
    # Whenever the number of trainin batches is specified, use that quantity to train

    if conditions[1] is True:
        # The parameter "eval_batches" is not specified, set it as the size of the val dataset
        eval_batches = len(eval_dl)
    
    if n_epochs is not None and conditions[2] is True:
        # Number of epoch specified but not the "evaluate_every" parameter,
        # set as evaluate_every the number of batches of one epoch
        evaluate_every = len(train_dl)
        
    
    #if any(conditions) and n_epochs is None:
        # Error! specify all the combinations of parameters or the number of epochs!
    #    raise ValueError('Either n_epoch is not None or all train_batches, eval_batches and evaluate_every are not None')
    #if all(conditions) and n_epochs is not None:
        # At least one is None.
        # Number of epochs specified.
    #    train_batches = len(train_dl)*n_epochs
    #    eval_batches = len(eval_dl)
    #    evaluate_every = len(train_dl)
    #if not any(conditions) is True:
        # All are not None
        # Number of epochs specified
    #    train_batches = min(train_batches, len(train_dl)*n_epochs)

    #train_batches = len(train_dl)*n_epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if verbose > 0:
        print('Start training...')
        print('Number of batches of the whole dataset: {} train, {} val.'.format(len(train_dl), len(eval_dl)))
        print('Number of epochs: {}'.format(n_epochs if n_epochs is not None else 'undefined'))
        print('Total number of batches that will be used during training: {}'.format(train_batches))
        print('Number of batches for a single evaluation: {}'.format(eval_batches))
        print('Results are reported every {} batches'.format(evaluate_every))
        print('Model: {}'.format(type(model).__name__))
        print('Device detected: {}'.format(device))
    
    train_it = iter(train_dl)
    eval_it = iter(eval_dl)

    results_train = defaultdict(list)
    results_eval = defaultdict(list)
    train_results_batches = defaultdict(list)
    train_step = 0

    # Extract the first batch
    batch = next(train_it)

    if verbose > 1:
        metrics_names = ['loss'] + [metric for metric in metrics]
        tp = TablePrinter(metrics_names, 4)
        tp.print_header()

    while True:
        # Train the batch
        train_dict = infer_batch(model, criterion, optimizer, batch, 
                                 metrics=metrics, device=device, **kwargs)
        for key, value in train_dict.items():
            train_results_batches[key].append(value)
        
        train_step += 1

        #print(sched)
        
        if sched is not None:
            #sched.step(results_eval['accuracy_mean'][-1])
            #print(sched.get_last_lr())
            sched.step()
            #print(sched.get_last_lr())

        # Check if it is time to evaluate
        if (train_step % evaluate_every) == 0:
            for key, value in train_results_batches.items():
                results_train[key + '_mean'].append(np.mean(value[-mean_lasts:]))
                results_train[key + '_std'].append(np.std(value[-mean_lasts:]))
            model.eval()
            eval_results_batches = defaultdict(list)
            with torch.no_grad():
                for _ in range(eval_batches):
                    try:
                        eval_batch = next(eval_it)
                    except StopIteration:
                        eval_it = iter(eval_dl)
                        eval_batch = next(eval_it)
                    eval_dict = infer_batch(model, criterion, optimizer, eval_batch,
                                            metrics=metrics, device=device, **kwargs)
                    for key, value in eval_dict.items():
                        eval_results_batches[key].append(value)
            for key, value in eval_results_batches.items():
                results_eval[key + '_mean'].append(np.mean(value))
                results_eval[key + '_std'].append(np.std(value))
            
            model.train()
            
            if verbose > 1:
                tp.print_line(results_train, results_eval)

        # Check if it is time to stop
        if train_step == train_batches:
            break

        # Extract a new training batch and repeat
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_dl)
            batch = next(train_it)
    return model, results_train, results_eval

def evaluate(model, criterion, eval_dl, 
             *, 
             metrics, eval_batches=None, **kwargs):
    
    if eval_batches is None:
        eval_batches = len(eval_dl)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('Start evaluating...')
    print('Number of batches: {}.'.format(len(eval_dl)))
    print('Select only the first {} batches'.format(eval_batches))
    print('Model: {}'.format(type(model).__name__))
    print('Device detected: {}'.format(device))
    
    classification_results = {'pair_id': [], 'prediction': [], 'label': []}

    model.eval()
    data = defaultdict(list)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dl), total=eval_batches):
            if i == eval_batches: break
            eval_dict, pair_id, pred, labels = infer_batch(model, criterion, None, batch,
                                    metrics=metrics, device=device,
                                    return_classification_results=True, **kwargs)
            
            classification_results['pair_id'].append(pair_id)
            classification_results['prediction'].append(pred)
            classification_results['label'].append(labels)

            for key, value in eval_dict.items():
                data[key].append(value)
    
    classification_results['pair_id'] = np.concatenate(classification_results['pair_id'])
    classification_results['prediction'] = np.concatenate(classification_results['prediction'])
    classification_results['label'] = np.concatenate(classification_results['label'])
    
    return data, pd.DataFrame(classification_results)

def infer_batch_test(model, batch,
                     *,
                     tokenizer, device):
    # Tokenize the inputs
    input_dict_pos = tokenize_inputs(batch[-2], tokenizer, device=device)
    input_dict_neg = tokenize_inputs(batch[-1], tokenizer, device=device)

    logits_pos = model(**input_dict_pos)
    #prob_pos = nn.functional.softmax(logits_pos, dim=1)[:, 0]

    #print(prob_pos)
    
    logits_neg = model(**input_dict_neg)
    #prob_neg = nn.functional.softmax(logits_neg, dim=1)[:, 0]

    #print(prob_neg)

    #scores = torch.stack([prob_pos, prob_neg], dim=1)
    #prob = nn.functional.softmax(scores, dim=1)

    #print(prob)
    
    #return prob
    return logits_pos, logits_neg

def evaluate_test(model, eval_dl,
                  *,
                  metrics, eval_batches=None, **kwargs):

    from sklearn.metrics import roc_auc_score
    
    if eval_batches is None:
        eval_batches = len(eval_dl)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('Start evaluating...')
    print('Number of batches: {}.'.format(len(eval_dl)))
    print('Select only the first {} batches'.format(eval_batches))
    print('Model: {}'.format(type(model).__name__))
    print('Device detected: {}'.format(device))

    model.eval()
    logits_pos_list = []
    logits_neg_list = []
    seqs_id_pos_list = []
    seqs_id_neg_list = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(eval_dl), total=eval_batches):
            heavy_id, light_id_pos, light_id_neg = batch[1], batch[2], batch[3]
            seqs_id_pos_list.append((heavy_id, light_id_pos))
            seqs_id_neg_list.append((heavy_id, light_id_neg))
            if i == eval_batches: break
            logits_pos, logits_neg = infer_batch_test(model, batch, device=device, **kwargs)
            logits_pos_list.append(logits_pos)
            logits_neg_list.append(logits_neg)
            
    heavy_id_pos_list = [x for x, _ in seqs_id_pos_list]
    heavy_id_pos_all = torch.concat(heavy_id_pos_list).cpu()
    
    light_id_pos_list = [x for _, x in seqs_id_pos_list]
    light_id_pos_all = torch.concat(light_id_pos_list).cpu()
    
    heavy_id_neg_list = [x for x, _ in seqs_id_neg_list]
    heavy_id_neg_all = torch.concat(heavy_id_neg_list).cpu()
    
    light_id_neg_list = [x for _, x in seqs_id_neg_list]
    light_id_neg_all = torch.concat(light_id_neg_list).cpu()
    
    logits_pos_list = [x[:, 0] for x in logits_pos_list]
    logits_pos_all = torch.concat(logits_pos_list).cpu()
    
    logits_neg_list = [x[:, 0] for x in logits_neg_list]
    logits_neg_all = torch.concat(logits_neg_list).cpu()

    labels_pos = torch.ones(len(logits_pos_all), dtype=int)
    labels_neg = torch.zeros(len(logits_neg_all), dtype=int)

    heavy_id = torch.concat([heavy_id_pos_all, heavy_id_neg_all])
    light_id = torch.concat([light_id_pos_all, light_id_neg_all])
    logits = torch.concat([logits_pos_all, logits_neg_all])
    labels = torch.concat([labels_pos, labels_neg])
        
    return pd.DataFrame({
        'heavy_id': heavy_id,
        'light_id': light_id,
        'logits': logits,
        'labels': labels
    })


        
        
            
