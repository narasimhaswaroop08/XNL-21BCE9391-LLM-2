import json
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def get_synonym(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        lemmas = [lemma.name() for syn in synonyms for lemma in syn.lemmas()]
        filtered = [lemma for lemma in set(lemmas) if lemma.lower() != word.lower()]
        if filtered:
            return random.choice(filtered)
    return word

def augment_text(text, prob=0.2):
    words = word_tokenize(text)
    new_words = [get_synonym(word) if random.random() < prob else word for word in words]
    return " ".join(new_words)

def augment_squad(input_file, output_file, prob=0.2):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            paragraph["context"] = augment_text(paragraph["context"], prob)
            for qa in paragraph["qas"]:
                qa["question"] = augment_text(qa["question"], prob)
                for answer in qa["answers"]:
                    answer["text"] = augment_text(answer["text"], prob)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

input_file = "train-v2.0.json"
output_file = "augmented_train-v2.0.json"
augment_squad(input_file, output_file)
