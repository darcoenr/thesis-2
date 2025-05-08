import train

MAIN_DATASET_DIRECTORY = '../datasets/new/classificator/germline_v'

parameters_dict = {
    'train_data_location': '{}/train_only_v/train_only_v.csv'.format(MAIN_DATASET_DIRECTORY),
    'val_data_location': '{}/val_only_v/val_only_v.csv'.format(MAIN_DATASET_DIRECTORY),
    'subsample': None,
    'frac': None,
    'shuffle': True,
    'seed': 3,
    'batch_size': 32,

    'model_name': 'average',
    'model_checkpoint': 'Exscientia/IgBert',
    'tokenizer_checkpoint': None,
    'number_of_layers': 3,
    'layer_size': 2048,

    'optimizer_name': 'adam',
    'lr': 1e-4,
    'beta1': 0.9,
    'beta2': 0.999,
    'momentum': 0,
    'weight_decay': 1e-4,

    'lr_scheduler_name': None,
    'start_factor': None,
    'end_factor': None,
    'total_iters': None,

    'n_epochs': 1,
    'train_batches': None,
    'eval_batches': 30,

    'evaluate_every': 100,
    'mean_lasts': 20,
    
    'save': True,
    'name': 'final_train_11'
}
train.train(**parameters_dict)