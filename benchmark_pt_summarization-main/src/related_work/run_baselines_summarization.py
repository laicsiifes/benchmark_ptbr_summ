import os

from src.corpora.corpora_utils import (read_cstnews_corpus, read_temario_corpus, read_xlsum_dataset,
                                       read_recognasumm_dataset, filter_dataset_empty_text)
from src.sentences_weighting.baselines import BaselinesEnum
from src.summarizers.baselines_summarizers import summarize_single_document, summarize_multi_document

from datasets import load_dataset


if __name__ == '__main__':

    # test_corpus_name = 'temario'
    test_corpus_name = 'cstnews'
    # test_corpus_name = 'recognasumm'
    # test_corpus_name = 'xlsum'

    corpus_dir = f'../../data/corpora/{test_corpus_name}'
    summaries_dir = f'../../data/summaries/{test_corpus_name}'

    baselines = list(BaselinesEnum)

    similarity_threshold = 0.0
    max_tokens_summary = 120

    is_filter_sentences = True
    is_save_json = False

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nCorpus: {test_corpus_name}\n')

    corpus = None
    summarize = None

    if test_corpus_name == 'cstnews':
        corpus = read_cstnews_corpus(corpus_dir)
        summarize = summarize_multi_document
    elif test_corpus_name == 'temario':
        corpus = read_temario_corpus(corpus_dir)
        summarize = summarize_single_document
    elif test_corpus_name == 'xlsum':
        dataset = load_dataset('csebuetnlp/xlsum', 'portuguese')
        dataset_filtered = filter_dataset_empty_text(dataset)
        corpus_dict = read_xlsum_dataset(dataset_filtered)
        corpus = corpus_dict['test']
        summarize = summarize_single_document
        is_save_json = True
    elif test_corpus_name == 'recognasumm':
        dataset = load_dataset('recogna-nlp/recognasumm')
        corpus_dict = read_recognasumm_dataset(dataset)
        corpus = corpus_dict['test']
        summarize = summarize_single_document
        is_save_json = True
    else:
        print(f'Error. corpus_name must be "cstnews" or "xlsum_pt".')
        exit(-1)

    summarize(corpus, similarity_threshold, baselines, max_tokens_summary, is_filter_sentences, summaries_dir,
              is_save_json)
