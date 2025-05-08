import argparse
import pandas as pd

from tqdm import tqdm

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location')
    parser.add_argument('-o', default='../datasets/paired.fasta')
    return parser

# ==========

# ===== FUNCTIONS

def generate_fasta(location, output_location, which='both'):
    df = pd.read_csv(location, index_col=0)
    with open(output_location, mode='w') as f:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            if which == 'both':
                concat_seq = row['heavy'] + row['light']
            else:
                concat_seq = row[which]
            f.write('>{}\n'.format(row['pair_id' if which == 'both' else '{}_id'.format(which)]))
            f.write('{}\n'.format(concat_seq))
    print('Saved: {}'.format(output_location))

# ==========

# ===== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    parser = generate_parser(parser)
    args = parser.parse_args()

    location = args.location
    output_location = args.o

    generate_fasta(location, output_location)

if __name__ == '__main__':
    main()