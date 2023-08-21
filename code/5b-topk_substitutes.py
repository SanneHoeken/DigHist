import torch, json, pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

def get_topk_substitutes(indexes, target_idx, model, tokenizer, k):

    mask_id = tokenizer.mask_token_id
    sent = [index for i, index in enumerate(indexes) if i not in target_idx[1:]]
    idx = target_idx[0]
    sent[idx] = mask_id
    inputs = torch.tensor([sent])

    with torch.no_grad():
        logits = model(inputs).logits

    softmax = torch.softmax(logits, dim=-1).squeeze_()
    topk_probs, topk_indices = torch.topk(softmax, k*100, sorted=True, dim=-1)
    topk_indices = torch.transpose(topk_indices, 0, 1)
    substitutes = [tokenizer.decode(ind[idx]) for ind in topk_indices]
    #topk_probs = torch.transpose(topk_probs, 0, 1)
    #probs = [prob[target_idx].item() for prob in topk_probs]
    
    final_subs = []
    for sub in substitutes:
        if len(final_subs) == k:
            break
        if all([len(sub) > 2,
                sub.isalpha(),
                sub not in stopWords]):
            final_subs.append(sub)
            
    return final_subs


def main(sent2encoding_path, target2usage_path, tokenizer_name, model_name, output_path, k=10):
     
    # get encoded sents
    with open(sent2encoding_path, 'r') as infile:
        sent2encoding = json.load(infile)

    with open(target2usage_path, 'rb') as infile:
        target2mentions = pickle.load(infile)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    vocab = tokenizer.vocab

    substitutes = dict()
    for t in tqdm(target2mentions):
        substitutes[t] = {w: 0 for w in vocab} #defaultdict(int)
        for sent_id, target_idx in zip(target2mentions[t]['sent_ids'], target2mentions[t]['target_idx']):
            sent = sent2encoding[sent_id]
            subs = get_topk_substitutes(sent, target_idx, model, tokenizer, k)
            
            for sub in subs:
                substitutes[t][sub] += 1

    with open(output_path, 'w') as outfile:
       json.dump(substitutes, outfile)


if __name__ == '__main__':

    tokenizer_name = 'emanjavacas/MacBERTh'
    model_name = 'emanjavacas/MacBERTh'
    k = 10

    timebin = '1530-1552'
    sent2encoding_path = f'../output/MacBERTh-encodings/{timebin}-sentid2encoding.json'
    target2usage_path = f'../output/usages/{timebin}_nouns2usages'
    output_path = f'../output/substitutes/{timebin}-top{k}substitutes.json'
    
    main(sent2encoding_path, target2usage_path, tokenizer_name, model_name, output_path, k=k)
    