import pickle, torch, csv
import pandas as pd

def project(input_path, dimension_path, output_path):
    
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

    output_df = pd.DataFrame(output)
    output_df.to_csv(output_path, index=False)


def proj_diff(projections_file1, projections_file2, output_path):

    projections1 = pd.read_csv(projections_file1)
    projections2 = pd.read_csv(projections_file2)

    mean_df1 = projections1.groupby('target').mean()['projection']
    mean_df2 = projections2.groupby('target').mean()['projection']
    
    results = [] 
    for target in mean_df1.keys():
        diff = mean_df2[target] - mean_df1[target]
        result = {'target': target, 'projdiff': diff}
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    

if __name__ == '__main__':
    
    timebins = ['1530-1552', '1580-1603'] # must be 2
    output_paths = [f'../output/results/{tb}_nouns_projections.csv' for tb in timebins]
    dimension_path = '../output/dimensions/dimension'
    
    for timebin, output_path in zip(timebins, output_paths):
        input_path = f'../output/usages/{timebin}_ALLnouns2usages'
        project(input_path, dimension_path, output_path)
    
    final_output_path = f'../output/results/projdiff-1530-1552_1580-1603_nouns.csv'
    proj_diff(output_paths[0], output_paths[1], final_output_path)