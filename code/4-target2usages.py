import torch, pickle, json, os
from transformers import AutoModel
from tqdm import tqdm

def find_target_mentions(sent2encoding, target2encoding):
    
    target2mentions = {t : {'sent_ids': [], 'target_idx': [], 'embeddings': []} for t in target2encoding}

    # iterate over sentences and find target word mentions
    print('Finding target word mentions...')
    n = 0
    for sent_id, sent_encodings in tqdm(sent2encoding.items()):
        if len(sent_encodings) > 512:
            n += 1
            continue
        for i in range(len(sent_encodings)):
            for t, target_ids in target2encoding.items():
                if sent_encodings[i:i+len(target_ids)] == target_ids:
                    target2mentions[t]['sent_ids'].append(sent_id)
                    target2mentions[t]['target_idx'].append((i, i+len(target_ids)))           
    print(f"Not included {n} sentences that were longer than 512 tokens.")
    return target2mentions


def extract_representations(sent2encoding, target2mentions, model_name, layer_selection):                    
    
        # load model
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        model.eval()

        # iterate over all sentence mentions of all target words
        for n, t in enumerate(target2mentions):
            print(f"Extracting {len(target2mentions[t]['sent_ids'])} representations of '{t}' (target {n+1} out of {len(target2mentions)})...")
            for i, sent_id in enumerate(tqdm(target2mentions[t]['sent_ids'])):

                # feed sentence encodings to the model    
                input_ids = torch.tensor([sent2encoding[sent_id]])
                encoded_layers = model(input_ids)[-1]
                
                # extract selection of hidden layer(s)
                if type(layer_selection) == int:
                    vecs = encoded_layers[layer_selection].squeeze(0)
                elif type(layer_selection) == list:
                    selected_encoded_layers = [encoded_layers[x] for x in layer_selection]
                    vecs = torch.mean(torch.stack(selected_encoded_layers), 0).squeeze(0)
                elif layer_selection == 'all':
                    vecs = torch.mean(torch.stack(encoded_layers), 0).squeeze(0)
                
                # target word selection 
                vecs = vecs.detach()
                start_idx, end_idx = target2mentions[t]['target_idx'][i]
                vecs = vecs[start_idx:end_idx]
                
                # aggregate sub-word embeddings (by averaging)
                vector = torch.mean(vecs, 0)
                if torch.isnan(vector).any():
                    print(t, sent_id, start_idx, end_idx)
                
                target2mentions[t]['embeddings'].append(vector)

        return target2mentions
        

def main(sent2encoding_path, target2encoding_path, target2usage_path, model_name, 
         layer_selection='all', find=True, extract=True):
     
    # get encoded sentences
    with open(sent2encoding_path, 'r') as infile:
        sent2encoding = json.load(infile)

    # get encoded targets
    with open(target2encoding_path, 'r') as infile:
        target2encoding = json.load(infile)

    # map target words to mentions in sentences
    if find:
        target2mentions = find_target_mentions(sent2encoding, target2encoding)
        with open(target2usage_path, 'wb') as outfile:
            pickle.dump(target2mentions, outfile)
    else:
        assert os.path.isfile(target2usage_path)
        with open(target2usage_path, 'rb') as infile:
            target2mentions = pickle.load(infile)

    # extract representations of target words mentions in sentences
    if extract:
        target2vectors = extract_representations(sent2encoding, target2mentions, model_name, layer_selection)
        with open(target2usage_path, 'wb') as outfile:
            pickle.dump(target2vectors, outfile)
        #for t in target2vectors:
            #print(t, len(target2vectors[t]['embeddings']))
    

if __name__ == '__main__':

    model_name = 'emanjavacas/MacBERTh'
    layer_selection = 'all'
    find = True
    extract = True

    timebin = '1580-1603'
    sent2encoding_path = f'../../../Data/DigHist/TimebinsProcessed/{timebin}-sentid2encoding.json' 
    target2encoding_path = f'../../../Data/DigHist/{timebin}_targets2encoding.json'
    target2usage_path = f'../../../Data/DigHist/{timebin}_targets2usages'

    main(sent2encoding_path, target2encoding_path, target2usage_path, model_name, 
         layer_selection=layer_selection, find=find, extract=extract)