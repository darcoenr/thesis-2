import torch
import pandas as pd

import utils

from torch.utils.data import DataLoader

test_data = '../datasets/new/test/val.csv'

df = pd.read_csv(test_data, index_col=0)

def evaluate(name,
             test_data, subsample, batch_size,
             model_name, model_checkpoint, tokenizer_checkpoint, eval_batches):

    location = r'../results/classification/{}/'.format(name)
    print('Location: {}'.format(location))
    
    # Get the data
    print('Retrieving the dataset...')
    test_ds, hyperpar = utils.get_test_dataset(test_data, subsample=subsample)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Load the model
    print('Load the model...')
    state_dict = torch.load('{}/model.pt'.format(location), weights_only=True)
    model = utils.load_model(model_name, checkpoint=model_checkpoint, state_dict=state_dict)

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if tokenizer_checkpoint is None: tokenizer_checkpoint = model_checkpoint
    tokenizer = utils.retrieve_tokenizer(tokenizer_checkpoint)

    logits = utils.evaluate_test(model, test_dl, metrics=None, eval_batches=None,
                                 tokenizer=tokenizer)

    logits.to_csv('{}/logits_val.csv'.format(location))

x = evaluate('final_train_11', 
             test_data, None, 32,
             'average', 'Exscientia/IgBert', None, 10000)

test_data = '../datasets/new/test/test.csv'

df = pd.read_csv(test_data, index_col=0)

def evaluate(name,
             test_data, subsample, batch_size,
             model_name, model_checkpoint, tokenizer_checkpoint, eval_batches):

    location = r'../results/classification/{}/'.format(name)
    print('Location: {}'.format(location))
    
    # Get the data
    print('Retrieving the dataset...')
    test_ds, hyperpar = utils.get_test_dataset(test_data, subsample=subsample)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Load the model
    print('Load the model...')
    state_dict = torch.load('{}/model.pt'.format(location), weights_only=True)
    model = utils.load_model(model_name, checkpoint=model_checkpoint, state_dict=state_dict)

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if tokenizer_checkpoint is None: tokenizer_checkpoint = model_checkpoint
    tokenizer = utils.retrieve_tokenizer(tokenizer_checkpoint)

    logits = utils.evaluate_test(model, test_dl, metrics=None, eval_batches=None,
                                 tokenizer=tokenizer)

    logits.to_csv('{}/logits_test.csv'.format(location))

x = evaluate('final_train_11', 
             test_data, None, 32,
             'average', 'Exscientia/IgBert', None, 10000)