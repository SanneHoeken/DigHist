import pandas as pd
import random, itertools

max_usages = 10
n_pairs = 10
t0_usages_path = '../data/1530-1552-testset-sentences.csv'
t1_usages_path = '../data/1580-1603-testset-sentences.csv'
output_file = '../data/1530-1552_1580-1603_testset-usage-pairsV2.csv'

# get lists of usage dicts
t0_usages = pd.read_csv(t0_usages_path)
t1_usages = pd.read_csv(t1_usages_path)
targets = set(t0_usages['target'])
t0_usages = t0_usages.to_dict(orient='records')
random.shuffle(t0_usages)
t1_usages = t1_usages.to_dict(orient='records')
random.shuffle(t1_usages)

# map [max_usages] usage dicts to targets 
t0_targets2records = {t:list() for t in targets}
t1_targets2records = {t:list() for t in targets}

for record in t0_usages:
    t = record['target']
    record['period'] = 't0'
    if len(t0_targets2records[t]) < max_usages:
        t0_targets2records[t].append(record)
    
for record in t1_usages:
    t = record['target']
    record['period'] = 't1'
    if len(t1_targets2records[t]) < max_usages:
        t1_targets2records[t].append(record)

merged_targets2records = {t:list() for t in targets}
for t in targets:
    merged_targets2records[t].extend(t0_targets2records[t])
    merged_targets2records[t].extend(t1_targets2records[t])

# sample random pairs of records and add to test instances
test_pairs = []
for t in targets:
    pairs = list(itertools.combinations(merged_targets2records[t], 2)) 
    n_pairs = n_pairs if n_pairs < len(pairs) else len(pairs)
    sample = random.sample(pairs, n_pairs)
    test_pairs.extend(sample)

test_instances = []
for pair in test_pairs:
    test_instance = {'target': pair[0]['target'],
                     'sent_id1': pair[0]['sent_id'],
                     'sentence1': pair[0]['sentence'],
                     'year1': pair[0]['year'],
                     'text1': pair[0]['text'],
                     'period1': pair[0]['period'],
                     'sent_id2': pair[1]['sent_id'],
                     'sentence2': pair[1]['sentence'],
                     'year2': pair[1]['year'],
                     'text2': pair[1]['text'],
                     'period2': pair[1]['period']}
    test_instances.append(test_instance)

test_df = pd.DataFrame(test_instances)
test_df.to_csv(output_file, index=False)

blind_df = test_df.drop(['year1', 'text1', 'period1', 
              'year2', 'text2', 'period2'], axis=1)
blind_df.to_csv(output_file.replace('.csv', '-BLIND.csv'), index=False)