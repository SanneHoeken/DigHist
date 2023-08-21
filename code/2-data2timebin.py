import glob, json
from pathlib import Path
import pandas as pd
from collections import defaultdict


def main(data_dir, output_dir):

    timebins = {'1485-1529': {'sentences': list(), 'counter': defaultdict(int), 'nouns': set()}, 
                '1530-1552': {'sentences': list(), 'counter': defaultdict(int), 'nouns': set()},
                '1553-1558': {'sentences': list(), 'counter': defaultdict(int), 'nouns': set()},
                '1559-1579': {'sentences': list(), 'counter': defaultdict(int), 'nouns': set()},
                '1580-1603': {'sentences': list(), 'counter': defaultdict(int), 'nouns': set()}}

    sent_id = 0

    for filepath in glob.iglob(f'{data_dir}/*-sentences.txt'):
        
        filename = Path(filepath).stem
        year = int(filename.split('_')[0])
        name = filename.split('_')[1].replace('-sentences', '')
        
        # determine timebin
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
        
        # store sentences in correct timebin
        with open(filepath, 'r') as infile:
            sentences = set([l.replace('\n', '') for l in infile.readlines()])
        for sent in sentences:
            sent_id += 1
            timebins[period]['sentences'].append({'id': sent_id, 'year': year, 'text': name, 'sentence': sent})

        # update counter of correct timebin
        with open(filepath.replace('-sentences.txt', '-counter.json'), 'r') as infile:
            counter = json.load(infile)
        for token, count in counter.items():
            timebins[period]['counter'][token] += count

        # update noun set of correct timebin
        with open(filepath.replace('-sentences.txt', '-nouns.txt'), 'r') as infile:
            nouns = set([l.replace('\n', '') for l in infile.readlines()])
        timebins[period]['nouns'].update(nouns)

    # save files per timebin
    for timebin, data in timebins.items():
        
        # save sentences (csv)
        sents_df = pd.DataFrame(data['sentences'])
        sents_df.to_csv(f'{output_dir}/{timebin}-sentencesV2.csv', index=False)
    
        # save nouns (one per line txt)
        with open(f'{output_dir}/{timebin}-nouns.txt', 'w') as outfile:
            for noun in data['nouns']:
                outfile.write(noun + '\n')

        # save token counter (json)
        with open(f'{output_dir}/{timebin}-counter.json', 'w') as outfile:
            json.dump(data['counter'], outfile, ensure_ascii=False)


if __name__ == '__main__':

    data_dir = '../../../Data/DigHist/ProcessedFiles'
    output_dir = '../../../Data/DigHist/TimebinsProcessed'

    main(data_dir, output_dir)