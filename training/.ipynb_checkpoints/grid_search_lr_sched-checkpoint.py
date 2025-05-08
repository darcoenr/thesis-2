import train

from itertools import product

MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/random'

combinations=[
    (None, None, None, None),
    ('linear', 1, 0.1,  1000),
    ('linear', 1, 0.05, 1000),
    ('linear', 1, 0.01, 1000),
    ('linear', 1, 0.1,  2000),
    ('linear', 1, 0.05, 2000),
    ('linear', 1, 0.01, 2000),
]

for i, (lr_sched_name, start_factor, end_factor, total_iters) in enumerate(combinations):
    parameters_dict = {
        'train_data_location': '{}/train_random/train.csv'.format(MAIN_DATASET_DIRECTORY),
        'val_data_location': '{}/val_random/val.csv'.format(MAIN_DATASET_DIRECTORY),
        'subsample': None,
        'frac': None,
        'shuffle': True,
        'seed': 999,
        'batch_size': 32,

        'model_name': 'average',
        'model_checkpoint': 'Exscientia/IgBert',
        'tokenizer_checkpoint': None,
        'number_of_layers': 3,
        'layer_size': 1024,

        'optimizer_name': 'adam',
        'lr': 1e-4,
        'momentum': 0,
        'weight_decay': 1e-4,

        'lr_scheduler_name': lr_sched_name,
        'start_factor': start_factor,
        'end_factor': end_factor,
        'total_iters': total_iters,

        'n_epochs': None,
        'train_batches': 10000,
        'eval_batches': 10,

        'evaluate_every': 50,
        'mean_lasts': 5,
    
        'save': True,
        'name': 'grid_search_16/{}'.format(i)
    }
    train.train(**parameters_dict)
