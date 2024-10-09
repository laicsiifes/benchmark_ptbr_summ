import sys

from src.corpora.corpora_utils import (read_cstnews_corpus, read_xsum_corpus, read_temario_corpus,
                                       read_recognasumm_dataset, read_summaries, read_summaries_json)
from datasets import load_dataset
from tqdm import tqdm
from src.basic_classes.document import Document
from src.evaluation_measures import rouge_parser

import spacy
import numpy as np

if __name__ == '__main__':

    # corpus_name = 'temario'
    # corpus_name = 'cstnews'
    corpus_name = 'recognasumm'
    # corpus_name = 'xlsum'

    corpus_path = f'../../data/corpora/{corpus_name}'
    summaries_dir = f'../../data/summaries/{corpus_name}'

    is_save_json = False
    test_corpus = None

    if corpus_name == 'cstnews':
        test_corpus = read_cstnews_corpus(corpus_path)
        documents = []
        for cluster_documents in test_corpus.list_cluster_documents:
            document = Document(name=cluster_documents.name.lower())
            document.references_summaries = cluster_documents.abstractive_reference_summaries
            documents.append(document)
        test_corpus.documents = documents
    elif corpus_name == 'temario':
        test_corpus = read_temario_corpus(corpus_path)
    elif corpus_name == 'xlsum_pt':
        corpus_dict = read_xsum_corpus(corpus_path)
        test_corpus = corpus_dict['test']
        is_save_json = True
    elif corpus_name == 'recognasumm':
        dataset = load_dataset('recogna-nlp/recognasumm')
        corpus_dict = read_recognasumm_dataset(dataset)
        test_corpus = corpus_dict['test']
        is_save_json = True
    else:
        print(f'Error. {corpus_name} INVALID!.')
        exit(-1)

    print(f'\n  Total documents: {len(test_corpus.documents)}\n')

    is_use_stemming = False
    limit_words = 100

    if is_save_json:
        read_summaries_json(test_corpus, summaries_dir)
    else:
        read_summaries(test_corpus, summaries_dir)

    nlp = spacy.load('pt_core_news_sm')

    def count_word(texto):
        doc = nlp(texto)
        return len([token for token in doc])
    
    for document in test_corpus.documents:
        for summary in document.candidate_summaries:
            total_words = [count_word(doc) for doc in test_corpus.documents]

            total_palavras = sum(total_words)
            media_palavras = np.mean(total_words)
            desvio_padrao_palavras = np.std(total_words)