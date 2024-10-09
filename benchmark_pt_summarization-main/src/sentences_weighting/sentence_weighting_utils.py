import networkx as nx


def filter_sentences(sentences: list):
    for i, sentence in enumerate(sentences):
        total_tokens = len(sentence.tokens)
        if not (10 <= total_tokens <= 70) or sentence.text.startswith('"') or sentence.text.startswith('-') or \
                sentence.text.startswith(';') or sentence.text.startswith('_'):
            sentence.is_removed = True


def build_sentences_graph(sentences: list, similarity_threshold: float) -> nx.Graph:

    sentences_graph = nx.Graph()

    for sentence in sentences:

        sentence_lemmas = set([t.lemma for t in sentence.tokens_no_stopwords])

        if len(sentence_lemmas) == 0:
            continue

        for other_sentence in sentences:

            if sentence.id_sentence != other_sentence.id_sentence:

                other_sentence_lemmas = set([t.lemma for t in other_sentence.tokens_no_stopwords])

                intersection = len(sentence_lemmas.intersection(other_sentence_lemmas))
                union = len(sentence_lemmas.union(other_sentence_lemmas))

                jaccard_similarity = intersection / union

                if jaccard_similarity > similarity_threshold:
                    sentences_graph.add_edge(sentence.id_sentence, other_sentence.id_sentence,
                                             weight=jaccard_similarity)

    return sentences_graph
