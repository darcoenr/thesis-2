import argparse
import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import BertModel
from tqdm import tqdm

import sys
sys.path.insert(0, '../training/')
import utils
import models

MODEL_CHECKPOINT = 'Exscientia/IgBert'
TOKENIZER_CHECKPOINT = 'Exscientia/IgBert'
MODEL_NAME = 'average'
BATCH_SIZE = 32

def extra_analysis(model_location, only_embeddings, dataset_location, additional_name):
    
    # Retrieve the needs
    print('Retrieving the model...')
    if only_embeddings:
        print('Return only the embeddings, use only the encoder ({}).'.format(MODEL_CHECKPOINT))
        model = BertModel.from_pretrained(MODEL_CHECKPOINT, add_pooling_layer=False)
    else:
        print('Return the logits, use the classificator ({}).'.format(model_location))
        state_dict = torch.load('{}'.format(model_location), weights_only=True)
        model = utils.load_model(MODEL_NAME, checkpoint=MODEL_CHECKPOINT, state_dict=state_dict)
    print('Retrieve the tokenizdr...')
    tokenizer = utils.retrieve_tokenizer(TOKENIZER_CHECKPOINT)
    print('Retrieve the dataset...')
    abd, _ = utils.get_dataset(dataset_location)
    data_dl = DataLoader(abd, batch_size=BATCH_SIZE)
    
    # Do the computation
    if only_embeddings:
        res = utils.compute_avg_embeddings(model, tokenizer, data_dl)
    else:
        # Ad hoc procedure
        res = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_dl), total=len(data_dl)):
                _, inputs, _ = batch[0], batch[1], batch[2]
                # Tokenize the inputs
                input_dict = utils.tokenize_inputs(inputs, tokenizer, device=device)
                # Compute model output
                logits = model(**input_dict)
                res.append(logits[:, 0].to('cpu'))
        res = torch.concat(res)
    
    # save
    if only_embeddings:
        which = 'embeddings'
    else:
        which = 'logits'
    name = which
    if additional_name != '':
        name = '{}-{}'.format(name, additional_name)
    name += '.pt'
    torch.save(res, name)
    
    
def generate_parser(parser):
    parser.add_argument('--model-location', default='../results/classification/final_train_1/model.pt')
    parser.add_argument('--only-embeddings', action='store_true')
    parser.add_argument('--dataset-location', default='sample.csv')
    parser.add_argument('--additional-name', default='')

def main():
    parser = argparse.ArgumentParser()
    generate_parser(parser)
    args = parser.parse_args()

    model_location = args.model_location
    only_embeddings = args.only_embeddings
    dataset_location = args.dataset_location
    additional_name = args.additional_name

    extra_analysis(model_location, only_embeddings, dataset_location, additional_name)
    

if __name__ == '__main__':
    main()