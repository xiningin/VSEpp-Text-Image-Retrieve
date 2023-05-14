import nltk
import pickle
from collections import Counter
import json
import argparse
import os
from tqdm import tqdm

"""
dataset.json中的数据形式：
    keys: ['images' , 'datset']

    images: {
        'sentids': [5615, 5616, 5617, 5618, 5619], 
        'imgid': 1123, 
        'sentences': [
            {'tokens': ['three', 'teenagers', 'are', 'carrying', 'wood', 'down', 'a', 'street', 'while', 'one', 'of', 'the', 'teenagers', 'is', 'smiling', 'at', 'the', 'camera'], 
            'raw': 'Three teenagers are carrying wood down a street while one of the teenagers is smiling at the camera.', 'imgid': 1123, 'sentid': 5615}, 
            {'tokens': ['a', 'man', 'in', 'a', 'blue', 'and', 'white', 't', 'shirt', 'trips', 'while', 'carrying', 'wood', 'with', 'another', 'man', 'in', 'a', 'white', 'jacket'], 
            'raw': 'A man in a blue and white t-shirt trips while carrying wood with another man in a white jacket.', 'imgid': 1123, 'sentid': 5616}, 
            {'tokens': ['men', 'are', 'struggling', 'to', 'carry', 'wood', 'down', 'the', 'street'], 
            'raw': 'Men are struggling to carry wood down the street.', 'imgid': 1123, 'sentid': 5617}, 
            {'tokens': ['these', 'three', 'boys', 'are', 'trying', 'to', 'carry', 'some', 'wood'], 
            'raw': 'These three boys are trying to carry some wood.', 'imgid': 1123, 'sentid': 5618}, 
            {'tokens': ['three', 'boys', 'carry', 'wooden', 'racks', 'down', 'a', 'sidewalk'], 
            'raw': 'Three boys carry wooden racks down a sidewalk.', 'imgid': 1123, 'sentid': 5619}
        ], 
        'split': 'train', 
        'filename': '1352231156.jpg'
    }

    dataset: 'flickr30K'
"""

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    

def from_flickr_json(path):
    dataset = json.load(open(path, 'r'))['images']
    captions = []
    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['sentences']]

    return captions

def build_vocab(data_path , data_name , threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter() # Used to count word frequency and discard word less than threshold
    full_path = os.path.join(data_path, data_name)
    captions = from_flickr_json(full_path)

    for i, caption in tqdm(enumerate(captions)):
        tokens = nltk.tokenize.word_tokenize(
            caption.lower()
        )
        counter.update(tokens) # Add to the counter

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(data_path, data_name):
    vocab = build_vocab(data_path , data_name , threshold=3)
    print(vocab.__len__())
    with open('../vocab/flickr30k_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", '../vocab/flickr30k_vocab.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--data_name', default='dataset.json')
    args = parser.parse_args()
    main(args.data_path, args.data_name)