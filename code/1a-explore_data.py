import glob, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

years = []
timebins = {'1485-1529': {'texts': 0, 'tokens': 0}, 
            '1530-1552': {'texts': 0, 'tokens': 0},
            '1553-1558': {'texts': 0, 'tokens': 0},
            '1559-1579': {'texts': 0, 'tokens': 0},
            '1580-1603': {'texts': 0, 'tokens': 0}}
year2tokens = {y: 0 for y in range(1485, 1604)}

data_dir = '../../../Data/DigHist/ProcessedData'

for filepath in tqdm(glob.iglob(f'{data_dir}/*.json')):
    
    filename = Path(filepath).stem
    year = int(filename.split('_')[0])
    
    with open(filepath, 'r') as infile:
        counter = json.load(infile)
    
    total_tokens = sum(list(counter.values()))
    year2tokens[year] += total_tokens
    years.append(year)

    # get stats per timebin
    if year < 1530:
        timebins['1485-1529']['texts'] += 1
        timebins['1485-1529']['tokens'] += total_tokens
    elif year < 1553:
        timebins['1530-1552']['texts'] += 1
        timebins['1530-1552']['tokens'] += total_tokens
    elif year < 1559:
        timebins['1553-1558']['texts'] += 1
        timebins['1553-1558']['tokens'] += total_tokens
    elif year < 1580:
        timebins['1559-1579']['texts'] += 1
        timebins['1559-1579']['tokens'] += total_tokens
    else:
        timebins['1580-1603']['texts'] += 1
        timebins['1580-1603']['tokens'] += total_tokens

print(timebins)

# PLOT TEXTS PER YEAR DISTRIBUTION
year_freq = Counter(years)
plt.bar(year_freq.keys(), year_freq.values())
plt.axvline(x = 1530, color='red')
plt.axvline(x = 1553, color='red')
plt.axvline(x = 1558, color='red')
plt.axvline(x = 1580, color='red')
plt.xlabel('Year')
plt.ylabel('Number of texts')
plt.show()

# PLOT TOKENS PER YEAR DISTRIBUTION
plt.bar(year2tokens.keys(), year2tokens.values())
plt.axvline(x = 1530, color='red')
plt.axvline(x = 1553, color='red')
plt.axvline(x = 1558, color='red')
plt.axvline(x = 1580, color='red')
plt.xlabel('Year')
plt.ylabel('Number of tokens')
plt.show()
