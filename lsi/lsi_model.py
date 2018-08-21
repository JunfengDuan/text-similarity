from gensim import models, similarities
import jieba
import pickle
import logging
from algorithm_library.gen_dict import gen_dict_corpus
from tfidf.tfidf_model import tfidf_save
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class LsiModel:

    def __init__(self, dictionary_path, corpus_path, raw_corpus_path, root_dir):
        self.dictionary_path = dictionary_path
        self.corpus_path = corpus_path
        self.raw_corpus_path = raw_corpus_path
        self.lsi_model_path = os.path.join(root_dir, 'lsi/lsi.model')
        self.tfidf_model_path = os.path.join(root_dir, 'lsi/tfidf.model')
        self.lsi_model = None
        self.similarity_lsi = None
        self.dictionary_lsi = None
        self.tfidf_model = None
        self.raw_corpus = None

        self.load_model()

    def lsi_save(self):

        tfidf_save(self.tfidf_model_path, self.corpus_path)

        with open(self.corpus_path, 'rb') as f1:
            corpus = pickle.load(f1)
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]

        with open(self.dictionary_path, 'rb') as f2:
            dictionary = pickle.load(f2)

        # initialize an LSI transformation
        # num_topics: 聚类的空间维度，维度越高，聚类越精确，计算效率越低.默认200
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)

        lsi.save(self.lsi_model_path)

    def lsi_update(self, new_corpus_path):
        """
        LSI训练是独一无二的，我们可以随时继续“训练”，只需提供更多的训练文件。
        这通过在一个称为在线训练的过程中对底层模型进行增量更新来完成。
        由于这个特征，输入文档流甚至可能是无限的 - 只要在LSI新文档到达的同时保持供应，同时使用计算的转换模型作为只读的！
        :param new_corpus_path:
        :return:
        """
        with open(new_corpus_path, 'rb') as f:
            new_corpus = pickle.load(f)

        lsi = models.LsiModel.load(self.lsi_model_path)
        lsi.add_documents(new_corpus)
        lsi.save(self.lsi_model_path)

    def load_model(self):

        with open(self.raw_corpus_path, 'r', encoding='utf-8') as f0:
            self.raw_corpus = f0.readlines()

        with open(self.dictionary_path, 'rb') as f:
            self.dictionary_lsi = pickle.load(f)

        self.tfidf_model = models.TfidfModel.load(self.tfidf_model_path)

        self.lsi_model = models.LsiModel.load(self.lsi_model_path)

        with open(self.corpus_path, 'rb') as f:
            corpus = pickle.load(f)
            corpus_tfidf = self.tfidf_model[corpus]

        lsi = models.LsiModel(corpus_tfidf)
        corpus_lsi = lsi[corpus_tfidf]  # vectorize input corpus in BoW format

        self.similarity_lsi = similarities.Similarity('Similarity-LSI-Index', corpus_lsi, num_features=400)

    def lsi_similarity(self, text, num_best):

        cut_words = list(jieba.cut(text))  # 分词
        text_corpus = self.dictionary_lsi.doc2bow(cut_words)  # 转换成bow向量
        text_corpus_tfidf = self.tfidf_model[text_corpus]  # 计算tfidf值
        lsi_value = self.lsi_model[text_corpus_tfidf]  # 计算lsi值

        self.similarity_lsi.num_best = num_best

        predicts = self.similarity_lsi[lsi_value]

        lsi_simi = []

        for t in predicts:
            id, p = t[0], t[1]
            text = self.raw_corpus[id]
            if text:
                lsi_simi.append((text.replace('\n', ''), p))

        return lsi_simi


def lsi_similarity_realtime(text, docs, num_best=10):
    dictionary, corpus = gen_dict_corpus(docs)

    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
    corpus_lsi = lsi[corpus_tfidf]

    similarity_lsi = similarities.Similarity('Similarity-LSI-Index', corpus_lsi, num_features=400,
                                             num_best=num_best)

    cut_words = list(jieba.cut(text))
    text_corpus = dictionary.doc2bow(cut_words)
    text_corpus_tfidf = tfidf[text_corpus]
    lsi_value = lsi[text_corpus_tfidf]

    return similarity_lsi[lsi_value]
