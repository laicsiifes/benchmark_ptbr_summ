import sys
import json
import os

from typing import Optional
from tqdm import tqdm

from src.basic_classes.corpus import Corpus
from src.basic_classes.document import Document
from src.basic_classes.summary import Summary
from src.analyzers import concept_analyzer as concept_analyzer
from src.analyzers import ilp_analyzer as ilp_analyzer
from src.nlp import spacy_nlp
from src.utils.utils import compute_sentence_similarity
from src.corpora.corpora_utils import save_summary


def summarize_cstnews(corpus: Corpus, min_ngram_size: int, max_ngram_size: int,
                      is_use_filtering: bool, weighting_method: str, summary_size: int,
                      summaries_dir: str, is_save_json: bool):

    with tqdm(total=len(corpus.list_cluster_documents), file=sys.stdout, colour='red', desc='Summarizing') as pbar:

        for cluster_documents in corpus.list_cluster_documents:

            all_concepts = []
            all_sentences = []

            for document in cluster_documents.documents:

                spacy_nlp.process_document(document)

                concept_analyzer.extract_concepts(document, min_ngram_size, max_ngram_size,
                                                  is_use_filtering)

                all_concepts.extend(document.concepts)
                all_sentences.extend(document.sentences)

            cluster_documents.concepts = all_concepts

            concept_analyzer.measure_concepts_weight_cluster(cluster_documents)

            for document in cluster_documents.documents:

                for sentence in document.sentences:
                    sentence.id_sentence = f'{document.id_document}_{sentence.id_sentence}'

                concept_analyzer.set_concepts_weight(document, weighting_method)

            filtered_sentences = []

            for sentence in all_sentences:
                if len(sentence.concepts) > 5:
                    filtered_sentences.append(sentence)

            selected_sentences = ilp_analyzer.select_sentences(filtered_sentences, summary_size)

            sentences_summary = []

            for sentence in selected_sentences:

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

            sentences_summary.sort(key=lambda s: s.id_sentence, reverse=False)

            text = '\n'.join([sentence.text for sentence in sentences_summary])
            text = text[:-1]

            summary = Summary(text=text, sentences=sentences_summary)

            summary.name = f'ilp_{weighting_method}_{min_ngram_size}_{max_ngram_size}'.lower()

            save_summary(cluster_documents.name, summary, summaries_dir)

            pbar.update(1)


def summarize_single_doc(corpus: Corpus, min_ngram_size: int, max_ngram_size: int,
                         is_use_filtering: bool, weighting_method: str, summary_size: int,
                         summaries_dir: str, is_save_json: bool):

    summary_name = f'{min_ngram_size}_{max_ngram_size}_{weighting_method}'

    list_documents_json = []

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='red', desc='Summarizing') as pbar:

        for document in corpus.documents:

            spacy_nlp.process_document(document)

            summary = summarize_document(document, min_ngram_size, max_ngram_size, is_use_filtering,
                                         weighting_method, summary_size)

            if summary is not None:
                summary.name = summary_name
                if is_save_json:
                    article_dict = {
                        'name': document.name,
                        'generated_summary': summary.text
                    }
                    list_documents_json.append(article_dict)
                    summary_path = os.path.join(summaries_dir, f'{summary_name}.json')
                    json_object = json.dumps(list_documents_json, indent=4)
                    os.makedirs(summaries_dir, exist_ok=True)
                    with open(file=summary_path, mode='w', encoding='utf-8') as outfile:
                        outfile.write(json_object)
                else:
                    save_summary(document.name, summary, summaries_dir)

            pbar.update(1)


def summarize_document(document: Document, min_ngram_size: int, max_ngram_size: int, is_use_filtering: bool,
                       weighting_method: str, summary_size: int) -> Optional[Summary]:

    concept_analyzer.extract_concepts(document, min_ngram_size, max_ngram_size, is_use_filtering)

    concept_analyzer.measure_concepts_weight_document(document)

    concept_analyzer.set_concepts_weight(document, weighting_method)

    filtered_sentences = []

    for sentence in document.sentences:
        if len(sentence.concepts) > 5:
            filtered_sentences.append(sentence)

    if len(filtered_sentences) <= 0:
        return None

    summary_sentences = ilp_analyzer.select_sentences(filtered_sentences, summary_size)

    if len(summary_sentences) <= 0:
        return None

    summary_sentences.sort(key=lambda s: s.id_sentence, reverse=False)

    text = '\n'.join([sentence.text for sentence in summary_sentences])
    text = text[:-1]

    summary = Summary(name=weighting_method, text=text, sentences=summary_sentences)

    return summary
