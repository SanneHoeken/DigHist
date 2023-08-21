import pickle

def main(target2usage_path, output_path, selection):

    filtered = {t : {'sent_ids': [], 'target_idx': [], 'embeddings': []} for t in selection}

    with open(target2usage_path, 'rb') as infile:
        t2u = pickle.load(infile)

    for t in t2u:
        for sent_id, idx in zip(t2u[t]['sent_ids'], t2u[t]['target_idx']):
            if int(sent_id) in selection[t] and sent_id not in filtered[t]['sent_ids']:
                filtered[t]['sent_ids'].append(sent_id)
                filtered[t]['target_idx'].append(idx)
        
    with open(output_path, 'wb') as outfile:
        pickle.dump(filtered, outfile)
        
if __name__ == '__main__':

    target2usage_path = '../output/usages/1580-1603_pairs2usages'
    output_path = '../output/usages/1580-1603_pairs2TENusages'

    selection = {'heretikes': [16746, 106817, 106915, 106874, 107662, 106597, 567292, 569474, 571724, 571173], 
                 'hipocrites': [464811, 464681, 523466, 551392, 553010, 688824, 688216, 36811, 35338, 32791],
                 'idolaters': [34784, 34492, 743901, 748112, 277702, 273104, 355398, 356958, 728924, 501336],
                 'papists': [32686, 597874, 646782, 611709, 645981, 726503, 729205, 628532, 693613, 693555],
                 'popelings': [744962, 598034, 597095, 728695, 728242, 728515, 557388, 728470, 597890, 553048],
                 'traitours': [596361, 706164, 705967, 743315, 745015, 550568, 553389, 430894, 481456, 552623],
                 'shavelings': [41792, 80087, 747575, 458332, 727614, 347700, 646350, 317500, 627113, 457715],
                 'harlots': [28811, 29281, 69091, 120544, 283327, 495823, 574224, 45379, 79189, 181041],
                 'strumpets': [44477, 149132, 278176, 188649, 342038, 342730, 492759, 80188, 725079, 785554],
                 'whores': [44986, 431667, 458550, 458727, 785387, 458490, 352013, 335298, 335125, 44685],
                 'catholikes': [1247, 16680, 16951, 705958, 553431, 706206, 706107, 17273, 17619, 451067],
                 'monkes': [17926, 17953, 149389, 81232, 151877, 152141, 150225, 150954, 189599, 422876],
                 'women': [14727, 17468, 17654, 69050, 708800, 712432, 726551, 742832, 13460, 744245],
                 'protestants': [13601, 14518, 14580, 14730, 14967, 552616, 706002, 742287, 729068, 228188]}

    main(target2usage_path, output_path, selection)