from scipy.spatial.distance import jensenshannon
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

def jensen_shannon_divergence(targets2subs_path1, targets2subs_path2, output_file):

    with open(targets2subs_path1, 'r') as infile:
        target2subs1 = json.load(infile)

    with open(targets2subs_path2, 'r') as infile:
        target2subs2 = json.load(infile)

    results = [] 
    for target in tqdm(target2subs1):
        vocab = set(target2subs1[target].keys())
        counts1 = np.array([target2subs1[target][sub] for sub in vocab])
        counts2 = np.array([target2subs2[target][sub] for sub in vocab])
        prob_dist1 = counts1 / counts1.sum()
        prob_dist2 = counts2 / counts2.sum()
        jsd = jensenshannon(prob_dist1, prob_dist2)
        result = {'target': target, 'jsd': jsd}
        results.append(result)
        
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    
    t2s_path1 = '../output/substitutes/1530-1552-top10substitutes.json'
    t2s_path2 = '../output/substitutes/1580-1603-top10substitutes.json'
    output_file = '../output/results/jsd-1530-1552_1580-1603_nouns.csv'
    
    jensen_shannon_divergence(t2s_path1, t2s_path2, output_file)