import re
import numpy as np
import networkx as nx
from heapq import nlargest
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from collections import Counter, defaultdict
from Normalization import Normalizer, _to_text, _WORD_PAT
from gensim.summarization.bm25 import get_bm25_weights


def summarize(text, P=5):
    sents = Normalizer(text).sent_tokenize()
    words = [Normalizer(sent).clean_up() for sent in sents]

    sim_mat = get_bm25_weights(words, n_jobs=-1)
    sim_mat_np = np.array(sim_mat)
    graph = nx.from_numpy_array(sim_mat_np)
    scores = nx.pagerank(graph)

    weighted = sorted(list(set(((scores[i], s) for i, s in enumerate(sents)))), reverse=True)[:P]

    return _to_text([tup[1] for tup in weighted]), len(words)


def naive_summarizer(text, P=5):
    ranking = defaultdict(int)
    sentances = Normalizer(text).sent_tokenize()
    words = [" ".join(Normalizer(sent).clean_up()) for sent in sentances]
    freq = Counter(words)

    for i, sent in enumerate(sentances):
        for w in re.findall(_WORD_PAT, sent):
            ranking[i] += freq[w]

    description_indecies = nlargest(P, ranking, key=ranking.get)

    return [sentances[i] for i in sorted(description_indecies)]


class Keywords:

    def __init__(self, text, window=2):
        self.text = text
        self.window = window
        self.words = Normalizer(text).clean_up()

    def _ngrams(self):
        ngramed = list()
        for i in range(len(self.words) - self.window + 1):
            ngramed.append(tuple(self.words[i:i + self.window]))
        return ngramed

    def _combination(self, ng):
        return list(set([(x, y) for x in ng for y in ng if x > y]))

    def extract(self):
        graph = nx.Graph()
        ngs = self._ngrams()
        for ng in ngs:
            edges = self._combination(ng)
            graph.add_edges_from(edges)
        scores = nx.pagerank(graph)
        return sorted(scores.items(), key= lambda sk: sk[1],reverse=True)


class TopicFinder:

    def __init__(self, texts):
        self.texts = Normalizer.clean_up(texts)

    def find_topic(self, num_topics, num_words=2,passes=20):
        dic = Dictionary(self.texts)
        corpus = [dic.doc2bow(text) for text in self.texts]
        lda = LdaModel(corpus, num_topics=num_topics, id2word=dic, passes=passes)
        return lda.top_topics(topn=2, dictionary=dic, corpus=corpus)
