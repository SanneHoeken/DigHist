import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score


def process_annotation_file(annos_file, data_file, second_file=False):

    second = '_second' if second_file else ''
    df_annotations = pd.read_csv(annos_file, sep=';')
    df_data = pd.read_csv(data_file)
    df_annotations = pd.merge(df_data, df_annotations)
    df_annotations = df_annotations.dropna()
    df_cannotdecide = df_annotations[(df_annotations['hate1'] == '-') | (df_annotations['hate2'] == '-') | (df_annotations['semantic_relatedness'] == '-')]
    print(f"Found {len(df_annotations)} annotated rows of which {len(df_cannotdecide)} with one or more 'Cannot decide' annotations")
    df_annotations = df_annotations[(df_annotations['hate1'] != '-') & (df_annotations['hate2'] != '-') & (df_annotations['semantic_relatedness'] != '-')]
    df_annotations['hate1'+second] = df_annotations['hate1'].astype(int)
    df_annotations['hate2'+second] = df_annotations['hate2'].astype(int)
    df_annotations['semantic_relatedness'+second] = df_annotations['semantic_relatedness'].astype(int)
    if second_file:
        df_annotations = df_annotations.drop(['hate1', 'hate2', 'semantic_relatedness'], axis=1)
    
    return df_annotations


def get_semantic_change_annotations(df_annotations):

    semantic_change_df = df_annotations.groupby(['target', 'period1', 'period2']).mean(numeric_only=True)['semantic_relatedness']
    target2change = {t: dict() for t in set(df_annotations['target'])}
    for t in semantic_change_df.keys():
        target = t[0]
        if t[1] == 't0' and t[2] == 't0':
            target2change[target]['earlier'] = semantic_change_df[t]
        elif t[1] == 't1' and t[2] == 't1':
            target2change[target]['later'] = semantic_change_df[t]
        elif t[1] == 't0' and t[2] == 't1':
            target2change[target]['compare'] = semantic_change_df[t]
        else:
            print(t)

    change_annotations = []
    for t in target2change:
        target2change[t]['target'] = t
        target2change[t]['delta_later'] = None
        if 'later' in target2change[t] and 'earlier' in target2change[t]:
            target2change[t]['delta_later'] = target2change[t]['later'] - target2change[t]['earlier']
        change_annotations.append(target2change[t])
    df_change_annotations = pd.DataFrame(change_annotations)

    return df_change_annotations


def get_hate_annotations(df_annotations):

    df_annos_unpaired1 = pd.DataFrame()
    df_annos_unpaired1['target'] = df_annotations['target']
    df_annos_unpaired1['sent_id'] = df_annotations['sent_id1']
    df_annos_unpaired1['hate'] = df_annotations['hate1']
    df_annos_unpaired1['period'] = df_annotations['period1']
    df_annos_unpaired2 = pd.DataFrame()
    df_annos_unpaired2['target'] = df_annotations['target']
    df_annos_unpaired2['sent_id'] = df_annotations['sent_id2']
    df_annos_unpaired2['hate'] = df_annotations['hate2']
    df_annos_unpaired2['period'] = df_annotations['period2']
    df_annos_unpaired = pd.concat([df_annos_unpaired1, df_annos_unpaired2])

    return df_annos_unpaired


def main(annos_files, data_file, jsd_file, projdiff_file, projections_file1, projections_file2):

    df_annotations = process_annotation_file(annos_files[0], data_file)

    # INTER-ANNOTATOR AGREEMENT
    if len(annos_files) > 1:
        df_annotations2 = process_annotation_file(annos_files[1], data_file, second_file=True)
        merged_annotations = pd.merge(df_annotations, df_annotations2)
        #merged_annotations.to_csv('../other/annotations_merged.csv')
        hate_column = pd.concat([merged_annotations['hate1'], merged_annotations['hate2']])
        hate_column_second = pd.concat([merged_annotations['hate1_second'], merged_annotations['hate2_second']])
        cohensk = cohen_kappa_score(hate_column, hate_column_second)
        corr, corr_p = pearsonr(hate_column, hate_column_second)
        print(f'Cohens Kappa for hate annotations (n = {len(hate_column)}): {round(cohensk, 3)}')
        print(f'Pearson correlation for hate annotations (n = {len(hate_column)}): {round(corr, 3)} (p = {round(corr_p, 3)})')
        change_column = merged_annotations['semantic_relatedness']
        change_column_second = merged_annotations['semantic_relatedness_second']
        cohensk = cohen_kappa_score(change_column,change_column_second)
        corr, corr_p = pearsonr(change_column, change_column_second)
        print(f'Cohens Kappa for semantic relatedness annotations (n = {len(change_column)}): {round(cohensk, 3)}')
        print(f'Pearson correlation for semantic relatedness annotations (n = {len(hate_column)}): {round(corr, 3)} (p = {round(corr_p, 3)})')
        merged_annotations['hate1'] = merged_annotations[['hate1', 'hate1_second']].mean(axis=1)
        merged_annotations['hate2'] = merged_annotations[['hate2', 'hate1_second']].mean(axis=1)
        merged_annotations['semantic_relatedness'] = merged_annotations[['semantic_relatedness', 'semantic_relatedness_second']].mean(axis=1)
        df_annotations = merged_annotations
              
    # SEMANTIC CHANGE METHOD
    df_change_annotations = get_semantic_change_annotations(df_annotations)
    df_jsd = pd.read_csv(jsd_file)
    change_merged_df = pd.merge(df_change_annotations, df_jsd)
    corr, corr_p = pearsonr(change_merged_df['compare'], change_merged_df['jsd'])
    n = len(change_merged_df['compare'])
    print(f'Pearson correlation between human COMPARE and JSD (n = {n}): {round(corr, 3)} (p = {round(corr_p, 3)})')

    # HATE PROJECTION METHOD
    df_annos_unpaired = get_hate_annotations(df_annotations)
    df_projections1 = pd.read_csv(projections_file1)
    df_projections1['period'] = ['t0' for x in range(len(df_projections1))]
    df_projections2 = pd.read_csv(projections_file2)
    df_projections2['period'] = ['t1' for x in range(len(df_projections2))]
    df_projections = pd.concat([df_projections1, df_projections2])
    hate_merged_df = pd.merge(df_annos_unpaired, df_projections).drop_duplicates()#.drop('sent_id', axis=1)
    corr, corr_p = pearsonr(hate_merged_df['hate'], hate_merged_df['projection'])
    n = len(hate_merged_df['hate'])
    print(f'Pearson correlation between human hate ratings and projection values (n = {n}): {round(corr, 3)} (p = {round(corr_p, 3)})')
    #hate_merged_df.to_csv('../other/hate_values_merged.csv')

    # HATE CHANGE
    """
    df_projdiff = pd.read_csv(projdiff_file)
    df_typehate = hate_merged_df.groupby(['target', 'period']).mean()['hate']
    hatediffs = []
    for t in df_typehate.keys():
        target = t[0]
        diff = df_typehate[(target, 't1')] - df_typehate[(target, 't0')] if (
            target, 't1') and (target, 't0') in df_typehate else None
        hatediffs.append({'target': target, 'hatediff': diff})
    df_hatediff = pd.DataFrame(hatediffs)
    hatechange_merged_df = pd.merge(df_hatediff, df_projdiff).drop_duplicates()
    corr, corr_p = pearsonr(hatechange_merged_df['hatediff'], hatechange_merged_df['projdiff'])
    n = len(hatechange_merged_df['hatediff'])
    print(f'Pearson correlation between human hate change ratings and projection difference values (n = {n}): {round(corr, 4)} (p = {round(corr_p, 4)})')
    """
    
    # MERGE ALL
    #all_merged_df = pd.merge(hatechange_merged_df, change_merged_df).set_index('target')
    typehate_merged_df = hate_merged_df.drop('sent_id', axis=1).groupby('target').mean(numeric_only=True)
    change_merged_df = change_merged_df.drop(['later', 'delta_later', 'earlier'], axis=1).set_index('target')
    all_merged_df = pd.merge(change_merged_df, typehate_merged_df, left_index=True, right_index=True)
    #all_merged_df.to_csv('../other/all_values_merged.csv')
    

if __name__ == '__main__':

    annos_files = ['../other/HHSCD_annotations_Sophie.csv', '../other/HHSCD_annotations500_Melvin.csv'] # max two 
    data_file = '../data/1530-1552_1580-1603_testset-usage-pairs.csv'
    jsd_file = '../output/results/jsd-1530-1552_1580-1603_nouns.csv'
    projdiff_file = '../output/results/projdiff-1530-1552_1580-1603_nouns.csv'
    projections_file1 = '../output/results/1530-1552_nouns_projections.csv'
    projections_file2 = '../output/results/1580-1603_nouns_projections.csv'

    main(annos_files, data_file, jsd_file, projdiff_file, projections_file1, projections_file2)