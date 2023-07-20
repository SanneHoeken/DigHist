import pickle

def main(target2usage_path, selection):

    filtered = {t : {'sent_ids': [], 'target_idx': [], 'embeddings': []} for t in selection}

    with open(target2usage_path, 'rb') as infile:
        t2u = pickle.load(infile)

    for t in t2u:
        for sent_id, idx, emb in zip(t2u[t]['sent_ids'], t2u[t]['target_idx'], t2u[t]['embeddings']):
            if sent_id in selection[t]:
                filtered[t]['sent_ids'].append(sent_id)
                filtered[t]['target_idx'].append(idx)
                filtered[t]['embeddings'].append(emb)
    
    with open(target2usage_path, 'wb') as outfile:
        pickle.dump(filtered, outfile)
        
if __name__ == '__main__':

    target2usage_path = f'../output/MacBERTh-encodings/1580-1603_pairs2usages'

    selection = {'heretikes': [], 
                 'hypocrites': [],
                 'idolaters': [],
                 'papists': [],
                 'popelings': [],
                 'romanists': [],
                 'shauelings': [],
                 'harlots': [],
                 'strumpets': [],
                 'whores': [],
                 'catholikes': [304, 7835, 7890, 346625, 352063, 346450, 346322, 269367, 262354, 8984],
                 'monkes': [8322, 8479, 38189, 34226, 35679, 36185, 36363, 40671, 41589, 177223],
                 'women': [2808, 7697, 9058, 171554, 173866, 173871, 176549, 111919, 99958, 84160]}

    main(target2usage_path, selection)