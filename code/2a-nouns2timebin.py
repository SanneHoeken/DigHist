import glob, json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

timebins = {'1485-1529': defaultdict(int), 
            '1530-1552': defaultdict(int),
            '1553-1558': defaultdict(int),
            '1559-1579': defaultdict(int),
            '1580-1603': defaultdict(int)}

min_freq = 10
max_freq = None
ending_with = 's'

data_dir = '../../../Data/DigHist/ProcessedData'
output_dir = '../../../Data/DigHist/TimebinsProcessed'

for filepath in tqdm(glob.iglob(f'{data_dir}/*-nouns.txt')):
    
    filename = Path(filepath).stem
    year = int(filename.split('_')[0])
    name = filename.split('_')[1].replace('-nouns', '')
    
    counter_filepath = f'{data_dir}/{year}_{name}-counter.json'
    with open(counter_filepath, 'r') as infile:
        counter = json.load(infile)

    with open(filepath, 'r') as infile:
        nouns = set([l.replace('\n', '') for l in infile.readlines()])
    
    # add nouns and their frequency to timebin
    if year < 1530:
        period = '1485-1529'
    elif year < 1553:
        period = '1530-1552'
    elif year < 1559:
        period = '1553-1558'
    elif year < 1580:
        period = '1559-1579'
    else:
        period = '1580-1603'
    
    for noun in nouns:
        timebins[period][noun] += counter[noun]


for period, noun_dic in timebins.items():
    
    noun_set = set(noun_dic.keys())
    
    # filter nouns that occur more then thres 
    if min_freq:
        noun_set = set([n for n in noun_set if noun_dic[n] >= min_freq])

    # filter nouns that occur less then thres 
    if max_freq:
        noun_set = set([n for n in noun_set if noun_dic[n] < max_freq])

    # filter for nouns ending with certain letter
    if ending_with:
        noun_set = set([n for n in noun_set if n.endswith(ending_with)])

    print(period, '\t', len(noun_set))

    with open(f'{output_dir}/{period}-filterednouns.txt', 'w') as outfile:
        for noun in noun_set:
            outfile.write(noun+'\n')
