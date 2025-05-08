import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

# Functions from germline_pairing.py

def clean(df):
    print('Cleaning...')
    informative_colums = [c for c in df.columns if 'call' in c and c != 'd_call_light']
    df = df[informative_colums + ['heavy', 'light']]
    def remove_allele(str):
        return str.split('*')[0]
    def remove_ig(str):
        return str[2:]
    def remove_gene(str):
        return str.split('-')[0]
    def remove_s(str):
        return str.split('S')[0]
    for c in informative_colums:
        df[c] = df[c].map(remove_allele).map(remove_ig).map(remove_gene).map(remove_s)
    return df

def generate_pairing_df(df):
    heavy = df['v_call_heavy'].unique()
    light = df['v_call_light'].unique()

    print('Number of unique heavy:', len(heavy))
    print('Number of unique light:', len(light))

    pairing_dict = {'heavy': [], 'light': [], 'cc': []}#, 'cn': [], 'nc': [], 'nn': []}
    prod = product(heavy, light)
    for h, l in tqdm(prod, total=len(heavy)*len(light)):
        pairing_dict['heavy'].append(h)
        pairing_dict['light'].append(l)
        cc = len(df[(df['v_call_heavy'] == h) & (df['v_call_light'] == l)])
        #cn = len(df[(df['v_call_heavy'] == h) & (df['v_call_light'] != l)])
        #nc = len(df[(df['v_call_heavy'] != h) & (df['v_call_light'] == l)])
        #nn = len(df[(df['v_call_heavy'] != h) & (df['v_call_light'] != l)])
        pairing_dict['cc'].append(cc)
        #pairing_dict['cn'].append(cn)
        #pairing_dict['nc'].append(nc)
        #pairing_dict['nn'].append(nn)

    df_pairing = pd.DataFrame(pairing_dict)
    return df_pairing

def main():
    # Read
    random_pairing = pd.read_csv('../datasets/negative_samples_random.csv', index_col=0)
    sequences = pd.read_csv('../datasets/sequences.csv', index_col=0)

    sequences = clean(sequences)
    pairing = generate_pairing_df(sequences)

    pairing = pairing[pairing['cc'] == 0]

    res = random_pairing.merge(sequences[['light', 'v_call_light']], how='left')
    res = res.merge(sequences[['heavy', 'v_call_heavy']], how='left')

    new_df = pd.merge(
        left=res, 
        right=pairing,
        how='left',
        left_on=['v_call_heavy', 'v_call_light'],
        right_on=['heavy', 'light'],
    )

    new_df = new_df[['heavy_x', 'light_x', 'cc']]
    new_df = new_df.rename({'heavy_x': 'heavy', 'light_x': 'light'}, axis=1)
    new_df.insert(len(new_df.columns), 'class', np.ones(len(new_df), dtype=np.int32))

    new_df[new_df['cc'] == 0][['heavy', 'light', 'class']].to_csv('../datasets/random_germline_0.csv')
    new_df[new_df['cc'] != 0][['heavy', 'light', 'class']].to_csv('../datasets/random_germline_not_0.csv')

if __name__ == '__main__':
    main()