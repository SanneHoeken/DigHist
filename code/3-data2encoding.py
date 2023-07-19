import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm


def encode_sentences(data_path, tknzr):

    data = pd.read_csv(data_path)
    sent_ids = data['id']
    sentences = data['sentence'].astype(str)

    encodings = [tknzr.encode(s.lower(), truncation=True) for s in tqdm(sentences)] #for batched processing: set padding='max_length'
    sent2encoding = {sent_id: encoding for sent_id, encoding in zip(sent_ids, encodings)}

    return sent2encoding


def encode_targets(targets_path, tknzr, pairs):

    if pairs:
        with open(targets_path, 'r') as infile:
            word_pairs = [tuple(x.replace('\n', '').split(';')) for x in infile.readlines()]
        targets = list(set([item for t in word_pairs for item in t]))

    else:
        with open(targets_path, 'r') as infile:
            targets = [x.replace('\n', '') for x in infile.readlines()]
        
    target2encoding = {t: tknzr.encode(t.lower(), add_special_tokens=False) for t in targets}

    return target2encoding


def main(input_path, output_path, model_name, datatype, pairs=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if datatype == 'sentences':
        encoding_dict = encode_sentences(input_path, tokenizer)
    elif datatype == 'targets':
        encoding_dict = encode_targets(input_path, tokenizer, pairs)

    with open(output_path, 'w') as outfile:
        json.dump(encoding_dict, outfile)


if __name__ == '__main__':
    
    timebin = '1580-1603'
    model_name = 'emanjavacas/MacBERTh'

    #datatype = 'sentences'
    #pairs = False
    #input_path = f'../../../Data/DigHist/TimebinsProcessed/{timebin}-sentences.csv'
    #output_path = f'../../../Data/DigHist/TimebinsProcessed/{timebin}-sentid2encoding.json'

    datatype = 'targets'
    pairs = False
    input_path = f'../../../Data/DigHist/{timebin}_targets.txt'
    output_path = f'../../../Data/DigHist/{timebin}_targets2encoding.json'
    
    main(input_path, output_path, model_name, datatype, pairs=pairs)
    