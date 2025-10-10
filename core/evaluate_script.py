import os
import evaluate

MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/germline_v'

MODELS_DIR = '../results/classification/grid_search_23'

dirs = os.listdir('{}'.format(MODELS_DIR))

for d in dirs:
    parameters_dict = {
        'name': 'grid_search_23/{}'.format(d),
    
        'val_data': '{}/val_only_v/val_only_v.csv'.format(MAIN_DATASET_DIRECTORY),
        'subsample': None,
        'frac': 0.15,
        'shuffle': True,
        'batch_size': 32,
        'seed': 0,
        'eval_batches': None,

        'model_name': 'average',
        'model_checkpoint': 'Exscientia/IgBert',
        'tokenizer_checkpoint': None,
    
        'save': True,
        'out': 'val_results'
    }
    evaluate.evaluate(**parameters_dict)

#MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/random'

#MODELS_DIR = '../results/classification/grid_search_18'

#dirs = os.listdir('{}'.format(MODELS_DIR))

#for d in dirs:
#    parameters_dict = {
#        'name': 'grid_search_18/{}'.format(d),
#    
#        'val_data': '{}/val_random/val.csv'.format(MAIN_DATASET_DIRECTORY),
#        'subsample': None,
#        'frac': 0.15,
#        'shuffle': True,
#        'batch_size': 32,
#        'seed': 0,
#        'eval_batches': None,

#        'model_name': 'average',
#        'model_checkpoint': 'Exscientia/IgBert',
#        'tokenizer_checkpoint': None,
    
#        'save': True,
#        'out': 'val_results'
#    }
#    evaluate.evaluate(**parameters_dict)

#MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/random'

#MODELS_DIR = '../results/classification/grid_search_19'

#dirs = os.listdir('{}'.format(MODELS_DIR))

#for d in dirs:
#    parameters_dict = {
#        'name': 'grid_search_19/{}'.format(d),
    
#        'val_data': '{}/val_random/val.csv'.format(MAIN_DATASET_DIRECTORY),
#        'subsample': None,
#        'frac': 0.15,
#        'shuffle': True,
#        'batch_size': 32,
#        'seed': 0,
#        'eval_batches': None,

#        'model_name': 'average',
#        'model_checkpoint': 'Exscientia/IgBert',
#        'tokenizer_checkpoint': None,
    
#        'save': True,
#        'out': 'val_results'
#    }
#    evaluate.evaluate(**parameters_dict)


