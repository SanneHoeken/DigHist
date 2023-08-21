import glob, spacy, re, json, os
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


def main(data_dir, output_dir, spelling_dict=None):
    
    nlp = spacy.load("en_core_web_sm")

    if spelling_dict:
        with open(spelling_dict, 'r') as infile:
            old2new = json.load(infile)

    for i, filepath in tqdm(enumerate(glob.iglob(f'{data_dir}/*.txt'))):
        
        nouns = []
        tokens = []
        header = True
        footer = False

        filename = Path(filepath).stem
        year = int(filename.split('_')[0])
        name = filename.split('_')[1].replace(' ', '')

        if not os.path.exists(f'{output_dir}/{year}_{name}-counter.json'):
            with open(filepath, 'r') as infile:
                content = infile.read()

            # remove urls, page references and lower text
            content = re.sub(r"http\S+", "", content)
            content = re.sub(r'(Page \d{1,5})', '', content)
            content = re.sub('Unnumbered page', '', content)
            content = content.replace('\n', '')
            content = content.lower()
            # content = re.sub("\[.*?\]", "", content) # remove text between square brackets
            
            sentences = sent_tokenize(content)
            filtered_sentences = []

            for sent in sentences:
                
                if 'volltext' in sent:
                    header = False
                if 'thema:' in sent or 'ustc-themenklassifizierung:' in sent:
                    footer = True

                if header == False and footer == False:
                    
                    # ignore sentences that contain no letters
                    if re.search('[a-zA-Z]', sent):
                        
                        # exclude one-word sentences
                        if len(sent.split()) > 1:
                            
                            if spelling_dict:
                                # apply spelling normalization and save tokens and nouns seperately
                                new_sent = []
                                doc = nlp(sent)
                                tokens_pos = [(token.text, token.pos_) for token in doc]
                                for token, pos in tokens_pos:
                                    if token in old2new:
                                        token = old2new[token]
                                    tokens.append(token)
                                    new_sent.append(token)
                                    if pos == 'NOUN':
                                        nouns.append(token)
                                new_sent = ' '.join(new_sent)
                            else:
                                new_sent = sent
                                # save tokens, and nouns seperate additionally 
                                doc = nlp(new_sent)
                                tokens_pos = [(token.text, token.pos_) for token in doc]
                                for token, pos in tokens_pos:
                                    tokens.append(token)
                                    if pos == 'NOUN':
                                        nouns.append(token)

                            filtered_sentences.append(new_sent)

            if header or not footer:
                print(filename)

            unique_nouns = set(nouns)
            token_counter = Counter(tokens)

            # save sentences (one per line txt)
            with open(f'{output_dir}/{year}_{name}-sentences.txt', 'w') as outfile:
                for sent in filtered_sentences:
                    outfile.write(sent + '\n')

            # save nouns (one per line txt)
            with open(f'{output_dir}/{year}_{name}-nouns.txt', 'w') as outfile:
                for noun in unique_nouns:
                    outfile.write(noun + '\n')

            # save token counter (json)
            with open(f'{output_dir}/{year}_{name}-counter.json', 'w') as outfile:
                json.dump(token_counter, outfile, ensure_ascii=False)

if __name__ == '__main__':

    data_dir = '../../../Data/DigHist/RawFiles'
    output_dir = '../../../Data/DigHist/ProcessedFiles'
    spelling_dict = '../data/spelling_dictionary.json'

    main(data_dir, output_dir, spelling_dict=spelling_dict)