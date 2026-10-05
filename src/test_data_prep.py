import json
from datasets import Dataset 
import nltk
from rapidfuzz import fuzz

nltk.download('punkt')

def load_dataset(path):
    with open(path, "r") as file:
        dataset = json.load(file)
    
    return dataset


def sentence_splitter(text):
    ''' 
    we use nltk to retrieve the sentences in the cleaned text.
    '''
    sentences = nltk.sent_tokenize(text)
    return sentences



def merge_sentences(sentences, size):
    ''' 
    In the following function we merge some sentences. This beacuse we
    want to avoid to have sentences that are too small. 
    We set the size to be of 30 characters, this will ensure that we'll have 
    senetcens of maximum length of 60 characters (the worst case, where we 
    merge too sentences of size 30 characters).
    '''
    i = 0
    new_sentences = []
    num_sentences = len(sentences)
    while i < num_sentences - 1:
        merged = False
        sent = sentences[i]
        if len(sent) < size:
            merged_sent = sent
            while len(merged_sent) < size and i < num_sentences - 1:
                merged_sent += sentences[i+1]
                i += 1
                merged = True
        if merged:
            new_sentences.append(merged_sent)
        else:
            new_sentences.append(sent)

        i += 1

    return new_sentences


def find_matches(cleaned_sentences, noisy_text):
    ''' 
    This function finds the best matching string for each given clean sentence(s).
    To do so we use the fuzzy ratio, that measures the similarity of two strings by
    computing the minimum number of single-character edits (insertions, delitions, or
    substitutions) required to tranform one string into the other.

    To do so we use an "exploration" window of size 400 (it is very large, but it is 
    useful to deal with some exeptionally long samples). First we set the size of the
    window equal to the size of the cleaned sample that we are considering, then we 
    incrementally increase the size of the window exploring each time a new character.
    Finally, we return the best matching.

    '''
    pairs = []
    j = 0
    next_index = 0
    for sentence in cleaned_sentences:
        # we start with a window that has the same size of the clean sentence we want to match
        
        window_size = len(sentence)
        if window_size - 5 > 0:
            window_size -= 5
    
        best_score = -1     # fuzzy score (we initialize it with the minimum value)
        best_match = ''        
        noisy_size = len(noisy_text)
        for _ in range(400):                # Explore the next 400 characters
            if window_size >= noisy_size:
                break
    
            candidate = noisy_text[j: j+window_size]

            score = fuzz.ratio(candidate, sentence)
            if score > best_score:
                best_score = score
                best_match = candidate
                next_index = window_size
            window_size += 1
        
        j += next_index
        pairs.append((best_match, sentence))

    
    return pairs


def main():
    # Load datasets
    noisy = load_dataset("../datasets/ocr_datasets/ita/original_ocr.json")
    cleaned = load_dataset("../datasets/ocr_datasets/ita/cleaned.json")

    # sort the keys of the noisy dictionary
    keys = noisy.keys()
    integer_keys = [int(key) for key in keys]
    integer_keys.sort()

    #training_dataset = {}
    #training_dataset['ocr'] = []
    #training_dataset['clean'] = []

    test_dataset = {}
    test_dataset['ocr'] = []
    test_dataset['clean'] = []

    test_ocr_samples = {}

    for key in integer_keys[:7]:
        sentences = sentence_splitter(cleaned[str(key)])    # split the dataset in sentences
        new_sentences = merge_sentences(sentences, size=30) # merge sentences that are shorter than 30 characters
        matches = find_matches(new_sentences, noisy[str(key)])  # match the clean samples with the corresponding ocr sample
        

        test_ocr_samples[key] = []

        for pair in matches:
            test_ocr_samples[key].append(pair[0])

    with open("../datasets/test_ocr_samples.json", "w", encoding='utf-8') as f:
        json.dump(test_ocr_samples, f, ensure_ascii=False, indent=4)

    return
    
        
main()