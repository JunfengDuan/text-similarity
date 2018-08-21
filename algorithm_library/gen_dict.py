from gensim import corpora
import jieba
import pickle
import os

# 首先将文本处理生成dictionary和corpus。
# dictionary是词典，包含词以及词在词典中对应的位置。
# corpus将文本存贮成(词在词典中位置，词频)这种形式，每个文本为一行。


def gen_dict_corpus(raw_documents):

    # 分词处理
    corpora_documents = [list(jieba.cut(item_text)) for item_text in raw_documents]

    # 词典，以(词，词频)方式存储
    dictionary = corpora.Dictionary(corpora_documents)

    # 词库，以(id，词频)方式存储
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]

    return dictionary, corpus


def dump_data(raw_docs, dictionary_path, corpus_path):

    if raw_docs:

        dictionary, corpus = gen_dict_corpus(raw_docs)

        with open(dictionary_path, 'wb') as f1:
            pickle.dump(dictionary, f1)

        with open(corpus_path, 'wb') as f2:
            pickle.dump(corpus, f2)


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


