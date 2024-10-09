from src.basic_classes.document import Document
from src.basic_classes.cluster_documents import ClusterDocuments
from src.utils.ngram_extractor import build_ngrams


def filter_concepts(concepts: list) -> list:
    new_concepts = []
    for concept in concepts:
        count_stop_words = 0
        for token in concept.sub_tokens:
            if token.is_stopword:
                count_stop_words += 1
        if count_stop_words != len(concept.sub_tokens):
            new_concepts.append(concept)
    return new_concepts


def extract_concepts(document: Document, min_ngram_size: int, max_ngram_size: int, is_use_filtering: bool):
    all_concepts = []
    for sentence in document.sentences:
        concepts = build_ngrams(sentence.tokens, min_ngram_size, max_ngram_size)
        if is_use_filtering:
            concepts = filter_concepts(concepts)
        if concepts:
            for id_concept, concept in enumerate(concepts, start=1):
                concept.id = id_concept
            sentence.concepts = concepts
            all_concepts.extend(concepts)
    document.concepts = all_concepts


def measure_concepts_weight_document(document: Document):
    for concept in document.concepts:
        sent_freq = 0
        sent_pos = 0
        for sentence in document.sentences:
            if concept in sentence.concepts:
                sent_freq += 1
        for sentence in document.sentences:
            if concept in sentence.concepts:
                sent_pos = 1 - ((sentence.id_sentence - 1) / (1.0 * len(document.sentences)))
                break
        combined_weight_sum = (sent_freq + sent_pos) / 2
        combined_weight_mult = sent_freq * sent_pos
        concept.add_weight('sent_freq', sent_freq)
        concept.add_weight('sent_pos', sent_pos)
        concept.add_weight('comb_sum', combined_weight_sum)
        concept.add_weight('comb_mult', combined_weight_mult)


def measure_concepts_weight_cluster(cluster_documents: ClusterDocuments):

    dic_highest_value = {
        'sent_freq': 0,
        'sent_pos': 0,
        'comb_sum': 0,
        'comb_mult': 0
    }

    dict_concepts = {}

    for concept in cluster_documents.concepts:
        if concept.lemma not in dict_concepts:
            dict_concepts[concept.lemma] = concept

    for lemma, concept in dict_concepts.items():

        document_frequency = 0
        sentence_position = 0

        for doc in cluster_documents.documents:
            if concept in doc.concepts:
                document_frequency += 1
            for sent in doc.sentences:
                if concept in sent.concepts:
                    sentence_position += 1 - ((sent.id_sentence - 1) / (1.0 * len(doc.sentences)))
                    break

        combined_weight_sum = (document_frequency + sentence_position) / 2
        combined_weight_mult = document_frequency * sentence_position

        concept.add_weight('sent_freq', document_frequency)
        concept.add_weight('sent_pos', sentence_position)
        concept.add_weight('comb_sum', combined_weight_sum)
        concept.add_weight('comb_mult', combined_weight_mult)

        if document_frequency > dic_highest_value['sent_freq']:
            dic_highest_value['sent_freq'] = document_frequency

        if sentence_position > dic_highest_value['sent_pos']:
            dic_highest_value['sent_pos'] = sentence_position

        if combined_weight_sum > dic_highest_value['comb_sum']:
            dic_highest_value['comb_sum'] = combined_weight_sum

        if combined_weight_mult > dic_highest_value['comb_mult']:
            dic_highest_value['comb_mult'] = combined_weight_mult

    for _, concept in dict_concepts.items():
        concept.weights['sent_freq'] /= dic_highest_value['sent_freq']
        concept.weights['sent_pos'] /= dic_highest_value['sent_pos']
        concept.weights['comb_sum'] /= dic_highest_value['comb_sum']
        concept.weights['comb_mult'] /= dic_highest_value['comb_mult']

    for concept in cluster_documents.concepts:
        concept.weights = dict_concepts[concept.lemma].weights


def set_concepts_weight(document: Document, method_label: str):
    highest_weight = 0
    for sentence in document.sentences:
        for concept in sentence.concepts:
            concept.weight = concept.weights[method_label]
            if concept.weight > highest_weight:
                highest_weight = concept.weight
    for sentence in document.sentences:
        for concept in sentence.concepts:
            concept.weight /= highest_weight
