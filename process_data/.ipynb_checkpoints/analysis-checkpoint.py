import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

import networkx as nx

from collections import defaultdict

DATASET_DIRECTORY = '../datasets'

df_germline = pd.read_csv('{}/germline_ids.csv'.format(DATASET_DIRECTORY), index_col=0)
print('Number of unique heavy germlines: {:4}'.format(df_germline['v_call_heavy'].nunique()))
print('Number of unique light germlines: {:4}'.format(df_germline['v_call_light'].nunique()))
print('Number of possible pairs:         {:4}'.format(len(df_germline)))
zero_occurrence_germline = df_germline[df_germline['counter'] == 0]
print('Number of never seen pairs:       {:4} ({:.2f}%)'.format(
    len(zero_occurrence_germline),
    len(zero_occurrence_germline)/len(df_germline) * 100))

heavy_germline = df_germline['v_call_heavy'].unique()
light_germline = df_germline['v_call_light'].unique()

coocc_matrix = np.zeros([len(heavy_germline), len(light_germline)])

for r, h in enumerate(heavy_germline):
    for c, l in enumerate(light_germline):
        # Log scale => log(n + 1), plus one to avoid log(0)
        coocc_matrix[r][c] = np.log(df_germline[
            (df_germline['v_call_heavy'] == h) &
            (df_germline['v_call_light'] == l)
        ]['counter'].iloc[0] + 1)

fig, ax = plt.subplots(1)

fig.set_size_inches(20, 10)

ax.set_yticks(range(len(heavy_germline)))
ax.set_yticklabels(heavy_germline)
ax.set_xticks(range(len(light_germline)))
ax.set_xticklabels(light_germline, rotation=90)

ax.imshow(coocc_matrix)

for r, h in enumerate(heavy_germline):
    for c, l in enumerate(light_germline):
        ax.text(c, r, df_germline[
            (df_germline['v_call_heavy'] == h) &
            (df_germline['v_call_light'] == l)
        ]['counter'].iloc[0],
        horizontalalignment='center',
        verticalalignment='center')
ax.set_title('Co-occurrence matrix.')

fig.savefig('analysis/co-occurrence.png')
print('Colors are in logaritmic scale.')

df_paired = pd.read_csv('{}/sequences.csv'.format(DATASET_DIRECTORY), index_col=0)
print('Number of paired sequences: {:6}'.format(len(df_paired)))
print('Number of unique heavy:     {:6}'.format(df_paired['heavy_id'].nunique()))
print('Number of unique light:     {:6}'.format(df_paired['light_id'].nunique()))
print('Number of unique pairs:     {:6}'.format(df_paired['pair_id'].nunique()))

def check_number_of_germlines(merged, which):
    id = which + '_id'
    germline = 'pairing_index' if which == 'pair' else 'v_call_' + which
    info = merged[[id, germline]].drop_duplicates()
    if merged[id].nunique() == len(info):
        print('Every {} has an unique germline'.format(which))
    else:
        print('There are {} sequences having more then one germline'.format(which))
        for seq, data in info.groupby(id):
            if(len(data) > 1):
                print('\tSequence {} has {} germlines ({}).'.format(seq, len(data), 
                                                                    [row[germline] 
                                                                     for _, row in data.iterrows()]))

merged = df_paired.merge(df_germline)
check_number_of_germlines(merged, 'heavy')
check_number_of_germlines(merged, 'light')
check_number_of_germlines(merged, 'pair')

heavy_germline_counter = {'germline': [], 'counter': []}
for germline, data in merged[['heavy', 'v_call_heavy']].groupby('v_call_heavy'):
    heavy_germline_counter['germline'].append(germline)
    heavy_germline_counter['counter'].append(len(data))
heavy_germline_counter = pd.DataFrame(heavy_germline_counter)
heavy_germline_counter = heavy_germline_counter.sort_values(by='counter', ascending=False)
heavy_germline_counter.reset_index(drop=True, inplace=True)

light_germline_counter = {'germline': [], 'counter': []}
for germline, data in merged[['light', 'v_call_light']].groupby('v_call_light'):
    light_germline_counter['germline'].append(germline)
    light_germline_counter['counter'].append(len(data))
light_germline_counter = pd.DataFrame(light_germline_counter)
light_germline_counter = light_germline_counter.sort_values(by='counter', ascending=False)
light_germline_counter.reset_index(drop=True, inplace=True)

fig, axs = plt.subplots(2)

axs[0].bar(range(len(heavy_germline_counter)), [np.log(x) for x in heavy_germline_counter['counter']])
axs[0].set_xticks(range(len(heavy_germline_counter)))
axs[0].set_xticklabels([x for x in heavy_germline_counter['germline']], rotation=90)

axs[1].bar(range(len(light_germline_counter)), [np.log(x) for x in light_germline_counter['counter']])
axs[1].set_xticks(range(len(light_germline_counter)))
axs[1].set_xticklabels([x for x in light_germline_counter['germline']], rotation=90)

fig.savefig('analysis/most_frequent.png')

print('Most frequent')

df_random_pairing = pd.read_csv('{}/negative_samples_random.csv'.format(DATASET_DIRECTORY), index_col=0)
print('Number of random pairing:        {}'.format(len(df_random_pairing)))
print('Number of unique random pairing: {}'.format(df_random_pairing['pair_id'].nunique()))

merged = pd.merge(df_random_pairing, df_germline)[['counter', 'heavy', 'light']]
zero_occurrence_germline = merged[merged['counter'] == 0]
print('Number of random pairing with zero germline: {}'.format(len(zero_occurrence_germline)))

df_germline_pairing = pd.read_csv('{}/negative_samples_germline.csv'.format(DATASET_DIRECTORY), index_col=0)
# Use x and y to avoid long line.
x = len(df_germline_pairing)
y = df_germline_pairing['pair_id'].nunique()
print('Number of germline pairing:        {}'.format(x))
print('Number of unique germline pairing: {} ({:.2f}%)'.format(y,
                                                               y / x * 100))

print('Number of zero germline pairs:      {}'.format(len(df_germline[df_germline['counter'] == 0])))
print('Number of different germline pairs: {}'.format(df_germline_pairing['pairing_index'].nunique()))

merged = df_germline_pairing.merge(df_germline)

coocc_matrix_2 = np.zeros([len(heavy_germline), len(light_germline)])

for r, h in enumerate(heavy_germline):
    for c, l in enumerate(light_germline):
        # Log scale => log(n + 1), plus one to avoid log(0)
        coocc_matrix_2[r][c] = np.log(len(merged[
            (merged['v_call_heavy'] == h) &
            (merged['v_call_light'] == l)
        ]) + 1)

fig, ax = plt.subplots(1)

fig.set_size_inches(20, 10)

ax.set_yticks(range(len(heavy_germline)))
ax.set_yticklabels(heavy_germline)
ax.set_xticks(range(len(light_germline)))
ax.set_xticklabels(light_germline, rotation=90)

ax.imshow(coocc_matrix_2)

for r, h in enumerate(heavy_germline):
    for c, l in enumerate(light_germline):
        ax.text(c, r, len(merged[
            (merged['v_call_heavy'] == h) &
            (merged['v_call_light'] == l)
        ]),
        horizontalalignment='center',
        verticalalignment='center')
ax.set_title('Co-occurrence matrix.')

fig.savefig('analysis/co-occurrence2.png')
print('Colors are in logaritmic scale.')

fig, ax = plt.subplots(1)
fig.set_size_inches(20, 10)

ax.set_yticks(range(len(heavy_germline)))
ax.set_yticklabels(heavy_germline)
ax.set_xticks(range(len(light_germline)))
ax.set_xticklabels(light_germline, rotation=90)

ax.imshow((coocc_matrix == 0).astype(int) + (coocc_matrix == coocc_matrix_2).astype(int))

fig.savefig('analysis/skipped.png')

def retrieve_data_and_cluster(data_name, cluster_name):
    data = pd.read_csv('{}/{}'.format(DATASET_DIRECTORY, data_name), index_col=0)
    cluster = pd.read_csv('{}/{}'.format(DATASET_DIRECTORY, cluster_name), sep='\t', 
                          header=None)
    cluster = cluster.rename({0: 'representative', 1: 'pair_id'}, axis=1)
    cluster = cluster.drop_duplicates()
    return data.merge(cluster)

def extract_cluster_info(cluster):
    print('Number of clusters:                      {:6}'.format(cluster['representative'].nunique()))
    cluster_sizes = {'representative': [], 'size': []}
    for representative, data in cluster.groupby('representative'):
        cluster_sizes['representative'].append(representative)
        cluster_sizes['size'].append(len(data))
    cluster_sizes = pd.DataFrame(cluster_sizes)
    print('Number of clusters with unique sequence: {:6}'.format(
        len(cluster_sizes[cluster_sizes['size'] == 1]))
    )
    print('Biggest cluster:                         {:6}'.format(max(cluster_sizes['size'])))
    values_count = cluster_sizes['size'].value_counts().map(lambda x: np.log(x))

    #fig, ax = plt.subplots(1)
    #ax.scatter(np.log(values_count.index), values_count)

    cluster_germlines = cluster[['representative', 'pairing_index']]
    cluster_germlines = cluster_germlines.merge(cluster_sizes[cluster_sizes['size'] > 1])
    all_the_same_germline = []
    which_ones = defaultdict(int)
    for representative, data in cluster_germlines.groupby('representative'):
        all_the_same_germline.append(data['pairing_index'].nunique() == 1)
        if data['pairing_index'].nunique() != 1:
            #which_ones.append((representative, sorted(data['pairing_index'].unique())))
            which_ones[frozenset(data['pairing_index'].unique())] += 1
    if all(all_the_same_germline):
        print('All the elements in the cluster have the same germline.')
    else:
        print('There are clusters whose elements have different germlines.')

        # Function copied from here:
        # https://www.tutorialspoint.com/finding-all-possible-pairs-in-a-list-using-python
        def find_all_pairs_optimized(lst):
            pairs = []
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    pairs.append((lst[i], lst[j]))
            return pairs

        g = nx.Graph()
        for s, _ in which_ones.items():
            s = list(s)
            g.add_edges_from(find_all_pairs_optimized(s))

        #fig, ax = plt.subplots(1)
        #nx.draw(g, node_size=10, ax=ax)

data_name = 'sequences.csv'
cluster_name = 'sequences.tsv'
paired_cluster = retrieve_data_and_cluster(data_name, cluster_name)
which_ones = extract_cluster_info(paired_cluster)

data_name = 'negative_samples_random.csv'
cluster_name = 'random.tsv'
random_cluster = retrieve_data_and_cluster(data_name, cluster_name)
extract_cluster_info(random_cluster)

data_name = 'negative_samples_germline.csv'
cluster_name = 'germline.tsv'
germline_cluster = retrieve_data_and_cluster(data_name, cluster_name)
extract_cluster_info(germline_cluster)

def read_splits(which):
    train = pd.read_csv('{}/train_{}.csv'.format(DATASET_DIRECTORY, which), index_col=0)
    val = pd.read_csv('{}/val_{}.csv'.format(DATASET_DIRECTORY, which), index_col=0)
    test = pd.read_csv('{}/test_{}.csv'.format(DATASET_DIRECTORY, which), index_col=0)
    return train, val, test

def extract_split_info(train, val, test, df_pos, df_neg):
    print('Train size:      {:6} ({} pos, {}, neg)'.format(len(train),
                                                           len(train[train['class'] == 0]),
                                                           len(train[train['class'] == 1])))
    print('Validation size: {:6} ({} pos, {}, neg)'.format(len(val),
                                                           len(val[val['class'] == 0]),
                                                           len(val[val['class'] == 1])))
    print('Test size:       {:6} ({} pos, {}, neg)'.format(len(test),
                                                           len(test[test['class'] == 0]),
                                                           len(test[test['class'] == 1])))
    
    train_pos = train[train['class'] == 0].merge(df_pos, how='left')
    train_neg = train[train['class'] == 1].merge(df_neg, how='left')
    print('Number of distinct germlines in train:     {}'.format(
        train_pos['pairing_index'].nunique() + train_neg['pairing_index'].nunique()))
    
    val_pos = val[val['class'] == 0].merge(df_pos, how='left')
    val_neg = val[val['class'] == 1].merge(df_neg, how='left')
    print('Number of distinct germlines in validaton: {}'.format(
        val_pos['pairing_index'].nunique() + val_neg['pairing_index'].nunique()))
    
    test_pos = test[test['class'] == 0].merge(df_pos, how='left')
    test_neg = test[test['class'] == 1].merge(df_neg, how='left')
    print('Number of distinct germlines in test:      {}'.format(
        test_pos['pairing_index'].nunique() + test_neg['pairing_index'].nunique()))
    
    train_germlines_pairs = set(
        list(train_pos['pairing_index'].unique()) + 
        list(train_neg['pairing_index'].unique())
    )

    valtest_germlines_pairs = set(
        list(val_pos['pairing_index'].unique()) + 
        list(val_neg['pairing_index'].unique()) +
        list(test_pos['pairing_index'].unique()) + 
        list(test_neg['pairing_index'].unique())
    )

    unique_to_train = train_germlines_pairs.difference(valtest_germlines_pairs)
    unique_to_valtest = valtest_germlines_pairs.difference(train_germlines_pairs)
    
    print('Number of germline pairs unique to training: {}'.format(
        len(unique_to_train))
    )
    print('Number of germline pairs unique to valtest:  {}'.format(
        len(unique_to_valtest))
    )

    number_uninque_to_train = 0
    for pairing_index in unique_to_train:
        number_uninque_to_train += (len(train_pos[train_pos['pairing_index'] == pairing_index]) +
                                    len(train_neg[train_neg['pairing_index'] == pairing_index]))
    print('Number of sequences whose germline is unique to training: {}'.format(
          number_uninque_to_train))
    
which = 'germline'
train, val, test = read_splits(which)
extract_split_info(train, val, test, df_paired, df_germline_pairing)