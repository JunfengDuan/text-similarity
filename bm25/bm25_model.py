from gensim.summarization import bm25
import jieba
import jieba.posseg as pseg
import os
from algorithm_library.gen_dict import write_file


class BM25Model:
    def __init__(self, dictionary_path, corpus_path):
        self.dictionary_path = dictionary_path
        self.corpus_path = corpus_path
        self.similarity_lsi = None
        self.dictionary_bm25 = None
        self.raw_corpus = None
        self.corpus = self.load_corpus()

    def bm25_similarity(self, text, num_best=10):
        query = list(jieba.cut(text))  # 分词

        bm = bm25.BM25(self.corpus)

        average_idf = sum(map(lambda k: float(bm.idf[k]), bm.idf.keys())) / len(bm.idf.keys())

        scores = bm.get_scores(query, average_idf)

        id_score = [(i, score) for i, score in enumerate(scores)]

        id_score.sort(key=lambda e: e[1], reverse=True)

        return id_score[0: num_best]

    def load_corpus(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            corpus = list(map(lambda l: l.replace('\n', '').split(' '), lines))
        return corpus


# 分词、去停用词
def gen_corpus(filename, corpus_path):
    stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            words = pseg.cut(line)
            key_words = [word for word, flag in words if flag not in stop_flag]
            result.append(' '.join(key_words)+'\n')
    write_file(result, corpus_path)
