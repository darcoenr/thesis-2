import os
import pathlib
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

import utils

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    # general
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--name', default='experiment')
    parser.add_argument('--save', action='store_true')
    # data hyperpar
    parser.add_argument('--train-data', default='../datasets/train_germline.csv')
    parser.add_argument('--val-data', default='../datasets/val_germline.csv')
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--subsample', type=int)
    parser.add_argument('--frac', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    # model hyperpar
    parser.add_argument('--model-name', default='average')
    parser.add_argument('--model-checkpoint', default='Exscientia/IgBert')
    parser.add_argument('--tokenizer-checkpoint')
    parser.add_argument('--number-of-layers')
    parser.add_argument('--layer-size')
    # optimizer hyperpar
    parser.add_argument('--optimizer-name', default='sgd')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    # epochs hyperpar  
    parser.add_argument('--n-epochs', type=int, default=1)
    parser.add_argument('--train-batches', type=int, default=None)
    parser.add_argument('--eval-batches', type=int, default=None)
    parser.add_argument('--evaluate-every', type=int, default=None),
    parser.add_argument('--mean-lasts', type=int, default=100)

    return parser

def train(
    # Data
    train_data_location='', val_data_location='', 
    shuffle=False, subsample=None, frac=None, seed=0,
    batch_size=32,          
    # Model
    model_name='', model_checkpoint='', tokenizer_checkpoint=None,
    number_of_layers=2, layer_size=1024,
    # Optimizer
    optimizer_name='', lr=0, 
    momentum=0, weight_decay=0,
    beta1=0.9, beta2=0.999, epsilon=1e-8,
    n_epochs=1, train_batches=None, eval_batches=None,
    evaluate_every=None, mean_lasts=None,
    # LR scheduler
    lr_scheduler_name=None,
    factor=None, patience=None, min_lr=None,
    gamma=None, last_epoch=-1,
    start_factor=1, end_factor=0.1, total_iters=100,
    # Saving
    save=False, name=''):
    
    hyperparameters = {}
    
    # Get the data
    print('Retrieving the datasets...')
    train_ds, train_ds_hyperpar = utils.get_dataset(train_data_location, shuffle=shuffle, subsample=subsample, frac=frac, seed=seed)
    val_ds, val_ds_hyperpar = utils.get_dataset(val_data_location, shuffle=shuffle, subsample=subsample, frac=frac, seed=seed)
    hyperparameters['train_dataset'] = train_ds_hyperpar
    hyperparameters['val_dataset'] = val_ds_hyperpar

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Get the model
    print('Retrieve the model...')
    model, model_hyperpar = utils.retrieve_model(model_name, 
                                                 checkpoint=model_checkpoint,
                                                 number_of_layers=number_of_layers, layer_size=layer_size)
    hyperparameters['model'] = model_hyperpar

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if tokenizer_checkpoint is None: tokenizer_checkpoint = model_checkpoint
    hyperparameters['tokenizer-checkpoint'] = tokenizer_checkpoint
    tokenizer = utils.retrieve_tokenizer(tokenizer_checkpoint)

    # Get the optimizer
    print('Retrieve the optimizer...')
    optimizer, optimizer_hyperpar = utils.retrieve_optimizer(optimizer_name, 
                                                             parameters=model.parameters(),
                                                             lr=lr,
                                                             momentum=momentum, weight_decay=weight_decay,
                                                             beta1=beta1, beta2=beta2, epsilon=epsilon)    
    hyperparameters['optimizer'] = optimizer_hyperpar

    # Get the LR scheduler
    if lr_scheduler_name is not None:
        sched, sched_hyperpar = utils.retrieve_lr_scheduler(lr_scheduler_name,
                                                            optimizer=optimizer, factor=factor,
                                                            patience=patience, min_lr=min_lr,
                                                            gamma=gamma, last_epoch=last_epoch,
                                                            start_factor=start_factor, end_factor=end_factor,
                                                            total_iters=total_iters)
        hyperparameters['lr_sched'] = sched_hyperpar
    else:
        sched = None

    hyperparameters['n_epochs'] = n_epochs
    hyperparameters['train_batches'] = train_batches
    hyperparameters['eval_batches'] = eval_batches
    hyperparameters['evaluate_every'] = evaluate_every
    hyperparameters['mean_last'] = mean_lasts

    criterion = nn.CrossEntropyLoss()

    model, results_train, results_eval = utils.train(
        model, criterion, optimizer, sched, train_dl, val_dl, 
        n_epochs=n_epochs,
        train_batches=train_batches, eval_batches=eval_batches, 
        evaluate_every=evaluate_every, mean_lasts=mean_lasts,
        tokenizer=tokenizer,
        metrics={'accuracy': accuracy_score}, verbose=2)
    
    if save:
        path = r'../results/classification/{}/'.format(name).strip()
        # Hyperparameters
        if not pathlib.Path(path).exists():
            os.makedirs(path)
        with open('{}/hyperpar.json'.format(path), 'w') as f:
            json.dump(hyperparameters, f)
        # Learning curves
        pd.DataFrame(results_train).to_csv('{}/results_train.csv'.format(path))
        pd.DataFrame(results_eval).to_csv('{}/results_eval.csv'.format(path))
        # Model
        state_dict = utils.get_state_dict(model_name, model)
        torch.save(state_dict, '{}/model.pt'.format(path))
    
# ==========

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)

    args = parser.parse_args()

    train_data_location = args.train_data
    val_data_location = args.val_data
    subsample = args.subsample
    batch_size = args.batch_size

    model_name = args.model_name
    model_checkpoint = args.model_checkpoint
    tokenizer_checkpoint = args.tokenizer_checkpoint
    number_of_layers = args.number_of_layers
    layer_size = args.layer_size

    optimizer_name = args.optimizer_name
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    n_epochs = args.n_epochs
    train_batches = args.train_batches
    eval_batches = args.eval_batches

    evaluate_every = args.evaluate_every
    mean_lasts = args.mean_lasts
    
    save = args.save
    name = args.name

    train(train_data_location, val_data_location, subsample, batch_size,
          model_name, model_checkpoint, tokenizer_checkpoint,
          optimizer_name, lr, momentum, weight_decay,
          n_epochs, train_batches, eval_batches,
          evaluate_every, mean_lasts,
          save, name)

    """
    hyperparameters = {}
    
    # Get the data
    print('Retrieving the datasets...')
    train_ds, train_ds_hyperpar = utils.get_dataset(args.train_data, subsample=args.subsample)
    val_ds, val_ds_hyperpar = utils.get_dataset(args.val_data, subsample=args.subsample)
    hyperparameters['train_dataset'] = train_ds_hyperpar
    hyperparameters['val_dataset'] = val_ds_hyperpar

    train_dl = DataLoader(train_ds, batch_size=args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    
    
    # Get the model
    print('Retrieve the model...')
    model, model_hyperpar = utils.retrieve_model(args.model_name, 
                                                 checkpoint=args.model_checkpoint)
    hyperparameters['model'] = model_hyperpar

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if args.tokenizer_checkpoint is None: args.tokenizer_checkpoint = args.model_checkpoint
    hyperparameters['tokenizer-checkpoint'] = args.tokenizer_checkpoint
    tokenizer = utils.retrieve_tokenizer(args.tokenizer_checkpoint)

    # Get the optimizer
    print('Retrieve the optimizer...')
    optimizer, optimizer_hyperpar = utils.retrieve_optimizer(args.optimizer_name, 
                                                             parameters=model.parameters(),
                                                             lr=args.lr,
                                                             momentum=args.momentum, weight_decay=args.weight_decay)
    hyperparameters['optimizer'] = optimizer_hyperpar

    hyperparameters['n_epochs'] = args.n_epochs
    hyperparameters['train_batches'] = args.train_batches
    hyperparameters['eval_batches'] = args.eval_batches
    hyperparameters['evaluate_every'] = args.evaluate_every
    hyperparameters['mean_last'] = args.mean_lasts

    criterion = nn.CrossEntropyLoss()

    model, results_train, results_eval = utils.train(
        model, criterion, optimizer, train_dl, val_dl, 
        n_epochs=args.n_epochs,
        train_batches=args.train_batches, eval_batches=args.eval_batches, 
        evaluate_every=args.evaluate_every, mean_lasts=args.mean_lasts,
        tokenizer=tokenizer,
        metrics={'accuracy': accuracy_score}, verbose=2)
    
    if save:
        path = r'../results/classification/{}/'.format(name).strip()
        # Hyperparameters
        if not pathlib.Path(path).exists():
            os.makedirs(path)
        with open('{}/hyperpar.json'.format(path), 'w') as f:
            json.dump(hyperparameters, f)
        # Learning curves
        pd.DataFrame(results_train).to_csv('{}/results_train.csv'.format(path))
        pd.DataFrame(results_eval).to_csv('{}/results_eval.csv'.format(path))
        # Model
        state_dict = utils.get_state_dict(args.model_name, model)
        torch.save(state_dict, '{}/model.pt'.format(path))
    """

if __name__ == '__main__':
    main()