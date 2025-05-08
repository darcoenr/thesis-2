import train

from itertools import product

MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/germline_v'

NUMBER_OF_LAYERS = [3, 5]
LAYER_SIZE = [1024, 2048]
LEARNING_RATE = [1e-3, 1e-4]
WEIGHT_DECAY = [1e-3, 1e-4, 1e-5]

combinations = list(product(NUMBER_OF_LAYERS, LAYER_SIZE, LEARNING_RATE, WEIGHT_DECAY))

START = 0
STOP = -1

for i, (number_of_layers, layer_size, learning_rate, weight_decay) in enumerate(combinations[START:STOP], START):
    parameters_dict = {
        'train_data_location': '{}/train_only_v/train_only_v.csv'.format(MAIN_DATASET_DIRECTORY),
        'val_data_location': '{}/val_only_v/val_only_v.csv'.format(MAIN_DATASET_DIRECTORY),
        'subsample': None,
        'frac': 0.05,
        'shuffle': True,
        'seed': 999,
        'batch_size': 32,

        'model_name': 'average',
        'model_checkpoint': 'Exscientia/IgBert',
        'tokenizer_checkpoint': None,
        'number_of_layers': number_of_layers,
        'layer_size':layer_size,

        'optimizer_name': 'adam',
        'lr': learning_rate,
        'momentum': 0,
        'weight_decay': weight_decay,

        'n_epochs': 3,
        'train_batches': None,
        'eval_batches': 10,

        'evaluate_every': 50,
        'mean_lasts': 5,
    
        'save': True,
        'name': 'grid_search_23/{}'.format(i)
    }
    train.train(**parameters_dict)
