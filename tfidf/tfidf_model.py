from algorithm_library.gen_dict import gen_dict_corpus
from gensim import corpora, models, similarities
import jieba
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_path = 'tfidf.model'
dictionary = None
tfidf_model = None
similarity_tfidf = None


def tfidf_save(tfidf_model_path, corpus_path):

    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    # initialize a model
    tfidf = models.TfidfModel(corpus)

    tfidf.save(tfidf_model_path)


def load_model():

    global tfidf_model
    global similarity_tfidf

    tfidf_model = models.TfidfModel.load(model_path)

    # Transforming vectors
    # 此时，tfidf被视为一个只读对象，可以用于将任何向量从旧表示（词频）转换为新表示（TfIdf实值权重）
    # doc_bow = [(0, 1), (1, 1)]
    # 使用模型tfidf，将doc_bow(由词,词频)表示转换成(词,tfidf)表示

    # 转换整个词库

    with open('../data/corpus.txt', 'rb') as f:
        corpus = pickle.load(f)
        corpus_tfidf = tfidf_model[corpus]

    similarity_tfidf = similarities.Similarity('Similarity-TFIDF-Index', corpus_tfidf, num_features=600)


def tfidf_similarity(text, num_best=10):

    global dictionary
    if not dictionary:
        with open('../data/dictionary.txt', 'rb') as f:
            dictionary = pickle.load(f)

    if not tfidf_model:
        load_model()

    cut_words = list(jieba.cut(text))  # ['北京', '雾', '霾', '红色', '预警']
    text_corpus = dictionary.doc2bow(cut_words)  # [(51, 1), (59, 1)]，即在字典的56和60的地方出现重复的字段，这个值可能会变化

    similarity_tfidf.num_best = num_best

    tfidf_value = tfidf_model[text_corpus]  # 根据之前训练生成的model，生成query的IFIDF值，然后进行相似度计算
    # [(51, 0.7071067811865475), (59, 0.7071067811865475)]
    return similarity_tfidf[tfidf_value]  # 返回最相似的样本材料,(index_of_document, similarity) tuples


if __name__ == "__main__":
    print()
    # tfidf_save()