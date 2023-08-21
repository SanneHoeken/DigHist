import glob, json, re

def get_modern_variant(word):

    word = word.replace('ā', 'an')
    word = word.replace('ū', 'un')
    word = word.replace('ē', 'en')
    word = word.replace('ā', 'am')
    word = word.replace('ū', 'um')
    word = word.replace('ē', 'em')
    word = re.sub("uy", r"vi", word)
    word = re.sub("([^q])u([aeiou])", r"\1v\2", word)
    word = word.replace('vv', 'w')
    word = re.sub(r"^vh", "wh", word)
    word = re.sub(r"v([bgnprstx])", r"u\1", word)
    word = re.sub(r"y", "i", word) if word != "i" else word
    word = re.sub(r"ie$", "y", word)
    word = re.sub("([aeiou])ie", r"\1y", word)
    word = re.sub(r"i$", "y", word) if word != "i" else word
    word = re.sub(r"^iou", "you", word)
    
    return word

def main(data_dir, output_file):
    
    vocab = set()
    old2new = dict()

    for filepath in glob.iglob(f'{data_dir}/*.json'):
        with open(filepath, 'r') as infile:
            counter = json.load(infile)
        for key in counter:
            vocab.add(key)

    for w in vocab:
        if w != get_modern_variant(w) and get_modern_variant(w) in vocab:
            old2new[w] = get_modern_variant(w)

    with open(output_file, 'w') as outfile:
        json.dump(old2new, outfile, ensure_ascii=False)


if __name__ == '__main__':

    data_dir = '../../../Data/DigHist/InitialProcessedFiles'
    output_file = '../data/spelling_dictionary.json'

    main(data_dir, output_file)