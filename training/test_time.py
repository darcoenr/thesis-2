import time
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertTokenizer

from utils import AbDataset
from models import AverageEmbedding, ClassificationHead

CHECKPOINT = 'Exscientia/IgBert'
HOW_MANY = 10

def main():
    df = pd.read_csv('../datasets/train_germline.csv')

    df = df.sample(1024)

    ds = AbDataset(df)

    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT, do_lower_case=False)
    bert_model = BertModel.from_pretrained(CHECKPOINT, add_pooling_layer=False).to('cuda')
    for p in bert_model.parameters():
        p.requires_grad = False
    mean_calculator = AverageEmbedding().to('cuda')
    classifier = ClassificationHead(1024).to('cuda')
    criterion = CrossEntropyLoss()

    time_dict = {
        'tokenize': [],
        'BERT': [],
        'mean': [],
        'classifier': [],
        'loss': []
    }

    t_start = time.time()

    for i, (seq, target) in enumerate(DataLoader(ds, batch_size=16, shuffle=True)):
        
        t1 = time.time()
        tokens = tokenizer(seq, add_special_tokens=True, padding=True, 
                           return_tensors='pt', return_special_tokens_mask=True)
        input_ids = tokens['input_ids'].to('cuda')
        attention_mask = tokens['attention_mask'].to('cuda')
        special_tokens_mask = tokens['special_tokens_mask'].to('cuda')

        t2 = time.time()
        embeddings = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        t3  = time.time()
        mean = mean_calculator(embeddings, special_tokens_mask)

        t4 = time.time()
        logits = classifier(mean)

        t5 = time.time()
        loss = criterion(logits, target.to('cuda'))
        loss.backward()

        t6 = time.time()
        time_dict['tokenize'].append(t2 - t1)
        time_dict['BERT'].append(t3 - t2)
        time_dict['mean'].append(t4 - t3)
        time_dict['classifier'].append(t5 - t4)
        time_dict['loss'].append(t6 - t5)

        for p in bert_model.parameters():
            if p.grad != None: print('NOT NONE')

    t_stop = time.time()

    for key, values in time_dict.items():
        print('{:12}{:6.4f}'.format(key + ':', np.mean(values)))

    print(t_stop - t_start)

if __name__ == '__main__':
    main()