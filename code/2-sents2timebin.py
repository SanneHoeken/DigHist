import glob
from pathlib import Path
import pandas as pd

timebins = {'1485-1529': [], 
            '1530-1552': [],
            '1553-1558': [],
            '1559-1579': [],
            '1580-1603': []}

data_dir = '../../../Data/DigHist/ProcessedData'
output_dir = '../../../Data/DigHist/TimebinsProcessed'

sent_id = 0

for filepath in glob.iglob(f'{data_dir}/*-sentences.txt'):
    
    filename = Path(filepath).stem
    year = int(filename.split('_')[0])
    name = filename.split('_')[1].replace('-sentences', '')

    with open(filepath, 'r') as infile:
        sentences = set([l.replace('\n', '') for l in infile.readlines()])
    
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
    
    for sent in sentences:
        sent_id += 1
        timebins[period].append({'id': sent_id, 'year': year, 'text': name, 'sentence': sent})

for timebin, data in timebins.items():
    data = pd.DataFrame(data)
    data.to_csv(f'{output_dir}/{timebin}-sentences.csv', index=False)


