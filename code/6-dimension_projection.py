import pickle, torch, csv

def main(input_path, dimension_path, output_path):
    
    cos = torch.nn.CosineSimilarity(dim=0)

    with open(input_path, 'rb') as infile:
        embeddings = pickle.load(infile)

    with open(dimension_path, 'rb') as infile:
        dimension_dic = pickle.load(infile)
        dimension = dimension_dic['dimension']

    output = []
    for test_word in embeddings:
        word_vecs = embeddings[test_word]['embeddings']
        sent_ids = embeddings[test_word]['sent_ids']
        for vec, sent_id in zip(word_vecs, sent_ids):
            cossim = cos(vec, dimension).item()
            output_dic = {'target': test_word, 'sent_id': sent_id, 'projection': cossim}
            output.append(output_dic)

    with open(output_path, 'w') as outfile:
        header = output[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=header)
        writer.writeheader()
        for dic in output:
            writer.writerow(dic)

if __name__ == '__main__':
    
    timebin = '1580-1603'
    input_path = f'../output/MacBERTh-encodings/{timebin}_targets2usages'
    dimension_path = f'../output/dimensions/dimension'
    output_path = f'../output/projections/{timebin}_target_projections.csv'
    
    main(input_path, dimension_path, output_path)