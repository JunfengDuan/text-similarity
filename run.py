import os
from algorithm_library.gen_dict import read_file, dump_data
from lsi.lsi_model import LsiModel, lsi_similarity_realtime

root_dir = os.path.dirname(__file__)
dictionary_path = os.path.join(root_dir, 'data/dictionary.txt')
corpus_path = os.path.join(root_dir, 'data/corpus.txt')
raw_corpus_path = os.path.join(root_dir, 'data/raw_corpus.txt')

lsi_model = None


def text_similarity(text, docs, num_best):
    return lsi_similarity_realtime(text, docs, num_best)


def text_similarity(text, num_best):
    """
    使用缓存的模型和文档集作近似计算
    :param text:
    :param num_best:
    :return:
    """
    global lsi_model
    if not lsi_model:
        lsi_model = LsiModel(dictionary_path, corpus_path, raw_corpus_path, root_dir)

    return lsi_model.lsi_similarity(text, num_best)


def cache_model_corpora():
    """
    缓存文档语料和模型到本地文件
    :return:
    """
    global lsi_model
    if not lsi_model:
        lsi_model = LsiModel(dictionary_path, corpus_path, raw_corpus_path, root_dir)

    raw_docs = read_file(raw_corpus_path)
    dump_data(raw_docs, dictionary_path, corpus_path)
    lsi_model.lsi_save()


def local_test():
    text = '女人爱美'
    docs = read_file(raw_corpus_path)
    predicts = text_similarity(text, docs, 5)
    for i, d in enumerate(docs):
        print(i, d)
    print('\n', predicts)


if __name__ == "__main__":
    print()
    # local_test()
    # cache_model_corpora()
    print(text_similarity('女人爱美', 5))