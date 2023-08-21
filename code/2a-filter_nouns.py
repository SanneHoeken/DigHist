import json

def main(data_dir, output_path, periods, to_exclude): 
    
    # filter criteria 
    min_freq = 10
    max_freq = None
    ending_with = 's' #'s' or None
    min_len = 3

    # get noun counter per period
    period2nouncounter = {}
    for period in periods:
        with open(f'{data_dir}/{period}-counter.json', 'r') as infile:
            tokencounter = json.load(infile)
        with open(f'{data_dir}/{period}-nouns.txt', 'r') as infile:
            nouns = set([l.replace('\n', '') for l in infile.readlines()])
        nouncounter = {noun: tokencounter[noun] for noun in nouns}
        period2nouncounter[period] = nouncounter

    # filter nouns
    period2nouns = dict()

    for period, noun_dic in period2nouncounter.items():
        
        noun_set = set(noun_dic.keys())
        
        # filter nouns that have letters only
        noun_set = set([n for n in noun_set if n.isalpha()])
        # filter nouns that occur more then thres 
        if min_freq:
            noun_set = set([n for n in noun_set if noun_dic[n] >= min_freq])
        # filter nouns that occur less then thres 
        if max_freq:
            noun_set = set([n for n in noun_set if noun_dic[n] < max_freq])
        # filter for nouns ending with certain letter
        if ending_with:
            noun_set = set([n for n in noun_set if n.endswith(ending_with)])
        # filter nouns that have at least min_len characters
        if min_len:
            noun_set = set([n for n in noun_set if len(n) > min_len])

        period2nouns[period] = noun_set
        print(period, '\t', len(noun_set))

    # Get intersection of multiple periods if needed
    if len(periods) > 1:
        noun_sets = [period2nouns[p] for p in periods]
        final_set = set.intersection(*noun_sets)
    elif len(periods) == 1:
        final_set = period2nouns[periods[0]]

    print('final set\t', len(final_set))

    # Save final set
    with open(output_path, 'w') as outfile:
        for noun in final_set:
            if noun not in to_exclude:
                outfile.write(noun+'\n')


if __name__ == '__main__':

    data_dir = '../../../Data/DigHist/TimebinsProcessed'
    #output_path = '../data/intersect-endingwiths-nouns_freq>=10-len>3.txt'
    #periods = ['1530-1552', '1580-1603']
    to_exclude = ['women', 'strumpets', 'whores', 'traitours', 'heretikes',
                  'hipocrites', 'harlots', 'shavelings', 'idolaters', 'popelings', 
                  'catholikes', 'papists', 'monkes', 'protestants']

    output_path = '../data/1580-1603-endingwiths-nouns_freq>=10-len>3.txt'
    periods = ['1580-1603']

    main(data_dir, output_path, periods, to_exclude)