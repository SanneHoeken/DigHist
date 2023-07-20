import pickle, csv
import pandas as pd

timebin = '1580-1603' #'1485-1529' #'1530-1552' #'1580-1603'
target2usage_path = f'../../../Data/DigHist/{timebin}_hateterms2usages'

with open(target2usage_path, 'rb') as infile:
    target2mentions = pickle.load(infile)

input_path = f'../../../Data/DigHist/TimebinsProcessed/{timebin}-sentences.csv'
with open(input_path, 'r') as infile:
    reader = csv.DictReader(infile)
    data = [dic for dic in reader]

output_data = []

for t in target2mentions:
    sent_ids = target2mentions[t]['sent_ids']
    for sent_id in sent_ids:
        for dic in data:
            if dic['id'] == sent_id:
                output_data.append({
                    'target': t, 
                    'sent_id': sent_id,
                    'sentence': dic['sentence'],
                    'year': dic['year'], 
                    'text': dic['text'], 
                })

output_data = pd.DataFrame(output_data)
output_data = output_data.drop_duplicates(keep='first')
output_data.to_csv(f'../../../Data/DigHist/{timebin}-hateterms-sentences.csv', index=False)