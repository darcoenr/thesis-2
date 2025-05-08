import train

from itertools import product

MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/random'

combinations=[
    (0.90, 0.999),
    (0.90, 0.980),
    (0.90, 0.960),
    (0.90, 0.940),
    (0.95, 0.999),
    (0.95, 0.980),
    (0.95, 0.960),
    (0.95, 0.940),
    (0.85, 0.999),
    (0.85, 0.980),
    (0.85, 0.960),
    (0.85, 0.940),
    (0.80, 0.999),
    (0.80, 0.980),
    (0.80, 0.960),
    (0.80, 0.940),
]

for i, (beta1, beta2) in enumerate(combinations):
    print(beta1, beta2)
    
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
        'beta1': beta1,
        'beta2': beta2,
        'momentum': 0,
        'weight_decay': 1e-3,

        'lr_scheduler_name': 'linear',
        'start_factor': 1,
        'end_factor': 0.05,
        'total_iters': 1000,

        'n_epochs': None,
        'train_batches': 10000,
        'eval_batches': 10,

        'evaluate_every': 50,
        'mean_lasts': 5,
    
        'save': True,
        'name': 'grid_search_19/{}'.format(i)
    }
    train.train(**parameters_dict)
