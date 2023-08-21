import pandas as pd
import random

result_files = ['../output/results/projdiff-1530-1552_1580-1603_nouns.csv',
                '../output/results/jsd-1530-1552_1580-1603_nouns.csv']
testset_file = '../data/testset.txt'

result_dfs = [pd.read_csv(f) for f in result_files]    
result_dfs = [df.set_index('target') for df in result_dfs]
merged_df = pd.concat(result_dfs, axis=1)
#print(merged_df.corr())

# test set creation: top / lowest value words + random sample
k = 20
topkprojdiff = set(merged_df['projdiff'].sort_values(ascending=False)[:k].keys())
lowkprojdiff = set(merged_df['projdiff'].sort_values()[:k].keys())
topkjsd = set(merged_df['jsd'].sort_values(ascending=False)[:k].keys())
lowkjsd = set(merged_df['jsd'].sort_values()[:k].keys())
testset = set.union(*[topkprojdiff, lowkprojdiff, topkjsd, lowkjsd])
testset.remove('verfügbarlinks')
rest = set(merged_df.index) - testset
random_sample = random.sample(list(rest), (5*k - len(testset)))
testset.update(random_sample)
test_df = merged_df.loc[list(testset)]

# write test set to file
with open(testset_file, 'w') as outfile:
    for w in testset:
        outfile.write(w+'\n')
