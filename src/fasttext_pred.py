import re
import numpy as np
import pickle
import fasttext


def pred(data, model):
    

    for d in data:
        if not "What is" in d['instruction'][0] or "What is(are)" in d['instruction'][0]:
            continue

        sentences = [txt.lower().strip() for txt in d['input'][0].split('\n')]
        keyword = extract_keyword(d['instruction'][0])
        sentence_vectors = [get_sentence_embedding(s, model) for s in sentences]
        keyword_vector = get_sentence_embedding(keyword, model)
        sims = [cos_sim(sentence_vector, keyword_vector) for sentence_vector in sentence_vectors]
        doc_vector = np.sum(sentence_vectors, axis=0)
        doc_sim = cos_sim(doc_vector, keyword_vector)
        
        if 'Sorry' in d['output'][0]:
            for i, s in enumerate(sentences):
                print("{} - {:<4}: {}".format(keyword, sims[i], s))
            print(f'doc sim {doc_sim}')
            print(d['output'][0])
            print('-----------------------------------')
            
            print('\n'*3)


def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words]

    if not word_vectors:
        return np.zeros(model.get_dimension())
    
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector


def extract_keyword(sentence):
    # Regular expression pattern to find the word(s) after "What is"
    pattern = r"What is\s+'([^']+)'"
    
    # Search for the pattern in the sentence
    match = re.search(pattern, sentence)
    
    if match:
        # Return the keyword found
        return match.group(1).strip()
    else:
        return None


def cos_sim(A, B):
  return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))


if __name__ == '__main__':
    data_path = '/home/junmin/Documents/dataset/mras/llm/processed_data/mras_en_otc_v3.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    model_path = 'output/otc_model_lower.bin'
    model = fasttext.load_model(model_path)

    data = data['alpaca_style']['train']
    pred(data, model)