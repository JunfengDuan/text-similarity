import os
from bm25.bm25_model import gen_corpus, BM25Model


root_dir = os.path.dirname(os.path.dirname(__file__))
dictionary_path = os.path.join(root_dir, 'data/bm25_corpus/dictionary.txt')
corpus_path = os.path.join(root_dir, 'data/bm25_corpus/corpus.txt')
raw_corpus_path = os.path.join(root_dir, 'data/raw_corpus.txt')


def text_similarity(text, top_num):
    bm25_model = BM25Model(dictionary_path, corpus_path)
    score = bm25_model.bm25_similarity(text, top_num)
    print(score)


if __name__ == "__main__":
    # gen_corpus(raw_corpus_path, corpus_path)
    print(text_similarity('女人爱美', 5))
