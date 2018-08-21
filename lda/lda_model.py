

def LDA():
    dictionary, corpus = gen_dict_corpus()
    corpus_tfidf = tf_idf()
    ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=2)

    ldamodel.print_topics()
    print(ldamodel.print_topics())