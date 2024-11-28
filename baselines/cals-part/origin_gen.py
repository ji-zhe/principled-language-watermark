# Key functions written by Wonhyk Ahn (whahnize@gmail.com)
# Refactoring and edits by KiYoon

import os
import random

from utils.dataset_utils import preprocess2sentence, get_dataset

random.seed(1230)
dtype='wikitext'
exp_name='origin'
if __name__ == "__main__":
    dirname = f"./results/context-ls/{dtype}/{exp_name}"
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    result_dir = os.path.join(dirname, "origin.txt")

    print("Processing datasets")
    _, corpus, num_sample2load = get_dataset(dtype)

    # cover_texts = preprocess_txt(corpus)
    cover_texts = [t.replace("\t", " ").replace("\n", " ") for t in corpus]
    cover_texts = preprocess2sentence(cover_texts, dtype+"-test", 0, num_sample2load['test'],
                                      spacy_model="en_core_web_sm",use_cache=False)
    
    assertion_wt2 = (dtype.startswith('wikitext2'))
    
    print("Start Writting")
    
    with open(result_dir,'w') as f:
        for sentences in cover_texts:
            for sentence in sentences:
                f.write(sentence.text.replace('<unk>','[UNK]').strip()+"\n")
        f.close()