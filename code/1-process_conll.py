import glob, spacy, re, json, os
from collections import Counter
from conllu import parse
from pathlib import Path
from tqdm import tqdm


def main(data_dir, output_dir):
    
    nlp = spacy.load("en_core_web_sm")

    for filepath in tqdm(glob.iglob(f'{data_dir}/*.conll')):
        
        nouns = []
        tokens = []

        filename = Path(filepath).stem
        year = int(filename.split('_')[0])
        name = filename.split('_')[1].replace(' ', '')

        with open(filepath, 'r') as infile:
            content = infile.read()

        parsed = parse(content)
        sentences = [' '.join([t['form'] for t in p]) for p in parsed]
        filtered_sentences = []

        for i, sent in enumerate(sentences):
            
            # remove page references and lower text
            sent = re.sub(r'(Page \d{1,5})', '', sent)
            sent = re.sub('Unnumbered page', '', sent)
            filtered_sentences.append(sent.lower())

            # save tokens, and nouns seperate additionally 
            doc = nlp(sent.lower())
            tokens_pos = [(token.text, token.pos_) for token in doc]
            for token, pos in tokens_pos:
                tokens.append(token)
                if pos == 'NOUN':
                    nouns.append(token)

        unique_nouns = set(nouns)
        token_counter = Counter(tokens)

        # save sentences (one per line txt)
        with open(f'{output_dir}/{year}_{name}-sentences.txt', 'w') as outfile:
            for sent in filtered_sentences:
                outfile.write(sent + '\n')

        # save nouns (one per line txt, alphabetically ordered)
        with open(f'{output_dir}/{year}_{name}-nouns.txt', 'w') as outfile:
            for noun in unique_nouns:
                outfile.write(noun + '\n')

        # save token counter (json)
        with open(f'{output_dir}/{year}_{name}-counter.json', 'w') as outfile:
            json.dump(token_counter, outfile, ensure_ascii=False)

if __name__ == '__main__':

    data_dir = '../../../Data/DigHist/RawCONLLFiles'
    output_dir = '../../../Data/DigHist/ProcessedCONLLFiles'

    main(data_dir, output_dir)