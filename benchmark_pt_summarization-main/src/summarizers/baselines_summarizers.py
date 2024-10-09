import sys
import json
import os

from tqdm import tqdm
from src.nlp import spacy_nlp
from src.basic_classes.corpus import Corpus
from src.basic_classes.document import Document
from src.basic_classes.summary import Summary
from src.basic_classes.cluster_documents import ClusterDocuments
from src.sentences_weighting.sentences_weighting_methods import (measure_sentence_weighting_single,
                                                                 measure_sentence_weighting_multi)
from src.sentences_weighting.sentence_weighting_utils import filter_sentences
from src.corpora.corpora_utils import save_summary
from src.utils.utils import compute_sentence_similarity


def summarize_single_document(corpus: Corpus, similarity_threshold: float, baselines: list,
                              max_tokens_summary: int, is_filter_sentences: bool, summaries_dir: str,
                              is_save_json: bool):

    for baseline in baselines:

        print(f'\n\n  Baseline: {baseline}\n')

        list_documents_json = []

        with tqdm(total=len(corpus.documents), file=sys.stdout, colour='red', desc='    Summarizing') as pbar:

            for document in corpus.documents:

                spacy_nlp.process_document(document)

                measure_sentence_weighting_single(document, similarity_threshold, baselines)

                summary = summarize_document_baselines(document, baseline.name.lower(), max_tokens_summary,
                                                       is_filter_sentences)

                if summary is not None:
                    if is_save_json:
                        article_dict = {
                            'name': document.name,
                            'generated_summary': summary.text
                        }
                        list_documents_json.append(article_dict)
                        summary_path = os.path.join(summaries_dir, f'{baseline.name.lower()}.json')
                        json_object = json.dumps(list_documents_json, indent=4)
                        os.makedirs(summaries_dir, exist_ok=True)
                        with open(file=summary_path, mode='w', encoding='utf-8') as outfile:
                            outfile.write(json_object)
                    else:
                        save_summary(document.name, summary, summaries_dir)

                pbar.update(1)


def summarize_multi_document(corpus: Corpus, similarity_threshold: float, baselines: list,
                             max_tokens_summary: int, is_filter_sentences: bool, summaries_dir: str,
                             is_save_json: bool):

    with tqdm(total=len(corpus.list_cluster_documents), file=sys.stdout, colour='red',
              desc='Summarizing') as pbar:

        for cluster_documents in corpus.list_cluster_documents:

            for document in cluster_documents.documents:

                spacy_nlp.process_document(document)

                for sentence in document.sentences:
                    sentence.full_id_sentence = f'{document.id_document}_{sentence.id_sentence}'

            measure_sentence_weighting_multi(cluster_documents, similarity_threshold, baselines)

            for baseline in baselines:

                summary = summarize_cluster_baselines(cluster_documents, baseline.name.lower(),
                                                      max_tokens_summary, is_filter_sentences)
                if summary is not None:
                    save_summary(cluster_documents.name, summary, summaries_dir)

            pbar.update(1)


def summarize_document_baselines(document: Document, baseline: str, max_tokens_summary: int,
                                 is_filter_sentences: bool):

    if is_filter_sentences:
        filter_sentences(document.sentences)

    for sentence in document.sentences:
        if not sentence.is_removed:
            if baseline in sentence.scores:
                sentence.relevance_score = sentence.scores[baseline]
            else:
                sentence.relevance_score = 0.0

    sorted_sentences = sorted(document.sentences, key=lambda s: s.relevance_score, reverse=True)

    summary_size = 0
    sentences_summary = []

    for sentence in sorted_sentences:

        if sentence.is_removed or sentence.relevance_score == 0.0:
            continue

        sentences_summary.append(sentence)
        summary_size += len(sentence.tokens)

        if summary_size >= max_tokens_summary:
            break

    if len(sentences_summary) == 0:

        sentences = sorted(document.sentences, key=lambda s: s.id_sentence)

        for sentence in sentences:

            if sentence.is_removed:
                continue

            sentences_summary.append(sentence)

            summary_size += len(sentence.tokens)

            if summary_size >= max_tokens_summary:
                break

    sentences_summary = sorted(sentences_summary, key=lambda s: s.id_sentence)

    summary_text = ''

    for sentence in sentences_summary:
        summary_text += f'{sentence.text}\n'

    summary_text = summary_text[:-1]
    summary = Summary(text=summary_text, name=baseline.lower())

    return summary


def summarize_cluster_baselines(cluster_documents: ClusterDocuments, baseline: str,
                                max_tokens_summary: int, is_filter_sentences: bool):

    if is_filter_sentences:
        for document in cluster_documents.documents:
            filter_sentences(document.sentences)

    all_sentences = []

    for document in cluster_documents.documents:
        for sentence in document.sentences:
            if not sentence.is_removed:
                if baseline in sentence.scores:
                    sentence.relevance_score = sentence.scores[baseline]
                else:
                    sentence.relevance_score = 0.0
                all_sentences.append(sentence)

    sorted_sentences = sorted(all_sentences, key=lambda s: s.relevance_score, reverse=True)

    summary_size = 0
    sentences_summary = []

    for sentence in sorted_sentences:

        if sentence.is_removed or sentence.relevance_score == 0.0:
            continue

        if sentence.text.endswith('\n'):
            sentence.text = sentence.text[:-1]

        has_similar_sentence = False

        for sent_summary in sentences_summary:

            similarity = compute_sentence_similarity(sentence, sent_summary)

            if similarity > 0.3:
                has_similar_sentence = True
                break

        if has_similar_sentence:
            continue

        sentences_summary.append(sentence)
        summary_size += len(sentence.tokens)

        if summary_size >= max_tokens_summary:
            break

    if len(sentences_summary) == 0:

        sentences = sorted(all_sentences, key=lambda s: s.full_id_sentence)

        for sentence in sentences:

            if sentence.is_removed:
                continue

            sentences_summary.append(sentence)

            summary_size += len(sentence.tokens)

            if summary_size >= max_tokens_summary:
                break

    sentences_summary = sorted(sentences_summary, key=lambda s: s.full_id_sentence)

    summary_text = ''

    for sentence in sentences_summary:
        summary_text += f'{sentence.text}\n'

    summary_text = summary_text[:-1]

    summary_text = summary_text.replace('\n\n', '\n')

    summary = Summary(text=summary_text, name=baseline.lower())

    return summary
