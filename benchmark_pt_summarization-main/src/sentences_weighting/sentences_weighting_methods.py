import math

from collections import Counter
from summa import keywords

from src.sentences_weighting.baselines import BaselinesEnum
from src.basic_classes.document import Document
from src.basic_classes.cluster_documents import ClusterDocuments
from src.sentences_weighting import sentence_weighting_utils as utils


def measure_sentence_weighting_single(document: Document, similarity_threshold: float, baselines: list):

    if BaselinesEnum.WORD_FREQUENCY in baselines:
        compute_word_frequency(document)

    if BaselinesEnum.TF_ISF in baselines:
        compute_tf_isf(document)

    if BaselinesEnum.SENTENCE_POSITION in baselines:
        compute_sentence_position(document)

    if BaselinesEnum.SENTENCE_CENTRALITY in baselines:
        compute_sentence_centrality(document)

    if BaselinesEnum.BUSHY_PATH in baselines:
        compute_bushy_path(document, similarity_threshold)

    if BaselinesEnum.AGGREGATE_SIMILARITY in baselines:
        compute_aggregate_similarity(document, similarity_threshold)

    if BaselinesEnum.NAMED_ENTITIES_FREQUENCY in baselines:
        compute_named_entities_frequency(document)

    if BaselinesEnum.TEXTRANK in baselines:
        compute_text_rank(document)


def measure_sentence_weighting_multi(cluster_documents: ClusterDocuments, similarity_threshold: float,
                                     baselines: list):

    unique_document = Document()

    unique_document.sentences = []
    unique_document.text = ''

    for document in cluster_documents.documents:
        unique_document.sentences.extend(document.sentences)
        unique_document.text += f'{document.text} '
        unique_document.add_tokens(document.tokens)
        unique_document.add_tokens_no_stopwords(document.tokens_no_stopwords)

    if BaselinesEnum.WORD_FREQUENCY in baselines:
        compute_word_frequency(unique_document)

    if BaselinesEnum.TF_ISF in baselines:
        compute_tf_isf(unique_document)

    if BaselinesEnum.SENTENCE_POSITION in baselines:
        compute_sentence_position_multi(cluster_documents)

    if BaselinesEnum.SENTENCE_CENTRALITY in baselines:
        compute_sentence_centrality(unique_document)

    if BaselinesEnum.BUSHY_PATH in baselines:
        compute_bushy_path(unique_document, similarity_threshold)

    if BaselinesEnum.AGGREGATE_SIMILARITY in baselines:
        compute_aggregate_similarity(unique_document, similarity_threshold)

    if BaselinesEnum.NAMED_ENTITIES_FREQUENCY in baselines:
        compute_named_entities_frequency(unique_document)

    if BaselinesEnum.TEXTRANK in baselines:
        compute_text_rank(unique_document)


def compute_word_frequency(document: Document):
    all_tokens = [token.lemma for token in document.tokens_no_stopwords]
    tokens_frequencies = Counter(all_tokens)
    highest_frequency = 0
    for sentence in document.sentences:
        score = 0
        for token in sentence.tokens_no_stopwords:
            score += tokens_frequencies[token.lemma]
        sentence.add_score(BaselinesEnum.WORD_FREQUENCY.name.lower(), score)
        if score > highest_frequency:
            highest_frequency = score
    if highest_frequency > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.WORD_FREQUENCY.name.lower()] /= highest_frequency


def compute_tf_isf(document: Document):

    all_tokens = [token.lemma for token in document.tokens_no_stopwords]

    sentences_lemmas = []

    for sentence in document.sentences:
        sentences_lemmas.append([t.lemma for t in sentence.tokens_no_stopwords])

    tokens_frequencies = Counter(all_tokens)

    distinct_tokens = set(all_tokens)

    sentences_frequency = {}

    for token_lemma in distinct_tokens:
        frequency = 0
        for sentence_lemmas in sentences_lemmas:
            if token_lemma in sentence_lemmas:
                frequency += 1
        sentences_frequency[token_lemma] = frequency

    highest_score = 0.0

    total_sentences = len(sentences_lemmas)

    for sentence in document.sentences:
        sentence_score = 0.0
        for token in sentence.tokens_no_stopwords:
            isf = total_sentences / sentences_frequency[token.lemma]
            tf_isf = math.log10(isf)
            tf_isf *= tokens_frequencies[token.lemma]
            sentence_score += tf_isf
        sentence.add_score(BaselinesEnum.TF_ISF.name.lower(), sentence_score)
        if sentence_score > highest_score:
            highest_score = sentence_score

    if highest_score > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.TF_ISF.name.lower()] /= highest_score


def compute_sentence_position(document: Document):
    total_sentences = len(document.sentences)
    for sentence in document.sentences:
        sentence_position = 1 - ((sentence.id_sentence - 1) / total_sentences)
        sentence.add_score(BaselinesEnum.SENTENCE_POSITION.name.lower(), sentence_position)


def compute_sentence_position_multi(cluster_documents: ClusterDocuments):
    for document in cluster_documents.documents:
        total_sentences = len(document.sentences)
        for sentence in document.sentences:
            sentence_position = 1 - ((sentence.id_sentence - 1) / total_sentences)
            sentence.add_score(BaselinesEnum.SENTENCE_POSITION.name.lower(), sentence_position)


def compute_sentence_centrality(document: Document):

    highest_score = 0.0

    for sentence in document.sentences:

        sentence_lemmas = set([t.lemma for t in sentence.tokens_no_stopwords])

        other_sentences_lemmas = []

        for other_sentence in document.sentences:
            if sentence.id_sentence != other_sentence.id_sentence:
                other_sentences_lemmas.extend([t.lemma for t in other_sentence.tokens_no_stopwords])

        other_sentences_lemmas = set(other_sentences_lemmas)

        intersection = len(sentence_lemmas.intersection(other_sentences_lemmas))
        union = len(sentence_lemmas.union(other_sentences_lemmas))

        sentence_score = intersection / union

        sentence.add_score(BaselinesEnum.SENTENCE_CENTRALITY.name.lower(), sentence_score)

        if sentence_score > highest_score:
            highest_score = sentence_score

    if highest_score > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.SENTENCE_CENTRALITY.name.lower()] /= highest_score


def compute_bushy_path(document: Document, similarity_threshold: float):

    highest_score = 0.0

    sentences_graph = utils.build_sentences_graph(document.sentences, similarity_threshold)

    for sentence in document.sentences:

        if not sentences_graph.has_node(sentence.id_sentence):
            sentence.add_score(BaselinesEnum.BUSHY_PATH.name.lower(), 0.0)
        else:
            edges = sentences_graph.edges(sentence.id_sentence)
            bushy_path_scores = len(edges)
            sentence.add_score(BaselinesEnum.BUSHY_PATH.name.lower(), bushy_path_scores)
            if bushy_path_scores > highest_score:
                highest_score = bushy_path_scores

    if highest_score > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.BUSHY_PATH.name.lower()] /= highest_score


def compute_aggregate_similarity(document: Document, similarity_threshold: float):

    highest_similarity = 0.0

    sentences_graph = utils.build_sentences_graph(document.sentences, similarity_threshold)

    for sentence in document.sentences:

        if not sentences_graph.has_node(sentence.id_sentence):
            sentence.add_score(BaselinesEnum.AGGREGATE_SIMILARITY.name.lower(), 0.0)
            continue

        list_edges = sentences_graph.edges(sentence.id_sentence)

        sentence_score = 0.0

        for edge in list_edges:
            sentence_score += sentences_graph.get_edge_data(edge[0], edge[1])['weight']

        sentence.add_score(BaselinesEnum.AGGREGATE_SIMILARITY.name.lower(), sentence_score)

        if sentence_score > highest_similarity:
            highest_similarity = sentence_score

    if highest_similarity > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.AGGREGATE_SIMILARITY.name.lower()] /= highest_similarity


def compute_named_entities_frequency(document: Document):
    highest_score = 0.0
    for sentence in document.sentences:
        sentence_score = len(sentence.entities)
        sentence.add_score(BaselinesEnum.NAMED_ENTITIES_FREQUENCY.name.lower(), sentence_score)
        if sentence_score > highest_score:
            highest_score = sentence_score
    if highest_score > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.NAMED_ENTITIES_FREQUENCY.name.lower()] /= highest_score


def compute_text_rank(document: Document):

    normalized_text = ''

    for sentence in document.sentences:
        sentence_text = ' '.join([t.lemma for t in sentence.tokens_no_stopwords])
        normalized_text += f'{sentence_text} '

    normalized_text = normalized_text.strip()

    document_keywords = keywords.keywords(normalized_text, ratio=1, language='portuguese', scores=True)

    document_keywords = dict(document_keywords)

    highest_score = 0

    for sentence in document.sentences:

        textrank_score = 0

        for token in sentence.tokens_no_stopwords:
            if token.lemma in document_keywords.keys():
                textrank_score += document_keywords[token.lemma]

        sentence.add_score(BaselinesEnum.TEXTRANK.name.lower(), textrank_score)

        if textrank_score > highest_score:
            highest_score = textrank_score

    if highest_score > 0:
        for sentence in document.sentences:
            sentence.scores[BaselinesEnum.TEXTRANK.name.lower()] /= highest_score
