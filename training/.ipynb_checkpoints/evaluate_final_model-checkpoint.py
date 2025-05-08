import os
import evaluate

MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/germline_v'

parameters_dict = {
    'name': 'final_train_11',
    
    'val_data': '{}/test_only_v/test_only_v.csv'.format(MAIN_DATASET_DIRECTORY),
    'subsample': None,
    'frac': None,
    'shuffle': True,
    'batch_size': 32,
    'seed': 0,
    'eval_batches': None,

    'model_name': 'average',
    'model_checkpoint': 'Exscientia/IgBert',
    'tokenizer_checkpoint': None,
    
    'save': True,
    'out': 'test_results'
}
evaluate.evaluate(**parameters_dict)