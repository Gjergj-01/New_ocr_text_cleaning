import json
from datasets import Dataset 
import nltk
from rapidfuzz import fuzz

nltk.download('punkt_tab')

def load_dataset(file_path):
    with open(file_path, "r") as file:
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
    want to avoid to have too small snetences. 
    We set the size to be of 100 characters, this will ensure that we'll have 
    senetcens of maximum length of 200 characters.
    '''
    i = 0
    new_sentences = []
    while i < len(sentences)-1:
        merged = False
        sent = sentences[i]
        if len(sent) < size:
            merged_sent = sent
            while len(merged_sent) < size and i < len(sentences)-1:
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
    This functions finds the best matching string for each given clean sentence(s).
    To do so we use the fuzzy ratio, that measures the similarity of two strings by
    computing the minimum number of single-character edits (insertions, delitions, or
    substitutions) required to tranform one string into the other.

    To do so we use an "exploration" window of size 85. First we set the size of the
    window equal to the size of the cleaned sample that we are considering, then we 
    incrementally increase the size of the window exploring each time a new character.

    '''
    pairs = []
    j = 0
    next_index = 0
    for sentence in cleaned_sentences:
        # we strat with a window that has the same size of the clean sentence we want to match
        window_size = len(sentence) 
        best_score = -1     # fuzzy score
        best_match = ''        
        noisy_size = len(noisy_text)
        for _ in range(150):
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
    noisy = load_dataset("../datasets/original_ocr.json")
    cleaned = load_dataset("../datasets/cleaned.json")

    keys = noisy.keys()
    integer_keys = [int(key) for key in keys]
    integer_keys.sort()

    training_dataset = {}
    training_dataset['ocr'] = []
    training_dataset['clean'] = []

    evaluation_dataset = {}
    evaluation_dataset['ocr'] = []
    evaluation_dataset['clean'] = []

    for key in integer_keys:
        sentences = sentence_splitter(cleaned[str(key)])
        new_sentences = merge_sentences(sentences, size=100)
        matches = find_matches(new_sentences, noisy[str(key)])

        for pair in matches:

            if key < 8:
                evaluation_dataset['ocr'].append(pair[0])
                evaluation_dataset['clean'].append(pair[1])
            else:
                training_dataset['ocr'].append(pair[0])
                training_dataset['clean'].append(pair[1])

    train = Dataset.from_dict(training_dataset)
    val = Dataset.from_dict(evaluation_dataset)

    train.save_to_disk("../datasets/t5-datasets/training.json")
    val.save_to_disk("../datasets/t5-datasets/validation.json")

    train.to_json("../datasets/t5-datasets/training_.json", force_ascii=False)
    val.to_json("../datasets/t5-datasets/validation_.json", force_ascii=False)


    return
    
        
main()