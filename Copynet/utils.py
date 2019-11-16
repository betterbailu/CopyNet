# some useful functions
# not related to this project

#from rouge import Rouge
from nltk.translate import bleu_score
from functools import reduce
import numpy as np

def generation_metrics(hypothesis_list, reference_list):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(
        hyps=list(reduce(lambda x,y:x+y,list(zip(*([hypothesis_list]*len(reference_list)))))),
        refs=reference_list * len(hypothesis_list),
        avg=True)
    bleu_scores = {}
    for i in range(1,6):
        bleu_scores['bleu%s'%i] = bleu_score.corpus_bleu(
            list_of_references=[reference_list]*len(hypothesis_list),
            hypotheses=hypothesis_list,
            weights=[1.0/i]*i)
    return rouge_scores, bleu_scores



def build_word_dict(src_path,tgt_path):
    vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    src_lines = open(src_path, 'r', encoding='utf-8').read().strip().split('\n')

    for line in src_lines:
        line = line.lower()
        words = line.split()
        for w in words:
            if w in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
            if w not in vocab:
                vocab.append(w)

    tgt_lines = open(tgt_path, 'r', encoding='utf-8').read().strip().split('\n')
    for line in tgt_lines:
        line = line.lower()
        words = line.split()
        for w in words:
            if w in ['PAD_TOKEN', 'UNKNOWN_TOKEN', 'START_DECODING', 'STOP_DECODING']:
                raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
            if w not in vocab:
                vocab.append(w)
    word_dict = {word: idx for idx, word in enumerate(vocab)}

    return word_dict


def build_emb_dict(glove_path, src_path, tgt_path):
    word_dict = build_word_dict(src_path, tgt_path)
    word2vec_dict = dict()
    with open(glove_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split(' ')
            if line[0] in word_dict:
                word2vec_dict[line[0]] = list(map(float, line[1:]))

    sorted_word = sorted(word_dict.items(), key=lambda x: x[1], reverse=False)

    vector = list(np.random.rand(100))
    embedding_matrix = []
    random_num = 0
    not_random_num = 0
    for (word, index) in sorted_word:
        if word in word2vec_dict:
            embedding_matrix.append(word2vec_dict[word])
            not_random_num+=1
        else:
            print(word)
            random_num += 1
            embedding_matrix.append(vector)
    return embedding_matrix
    # print(random_num)
    # print(not_random_num)

if __name__ == "__main__":
    build_emb_dict("C:/Users/t-linxli/Desktop/glove.twitter.27B.100d.txt","C:/Users/t-linxli/Desktop/out_rewrite.txt"
                   ,"C:/Users/t-linxli/Desktop/out_rewrite.txt")
