import os

from src.corpora.corpora_utils import (read_cstnews_corpus, read_temario_corpus, read_xlsum_dataset,
                                       read_recognasumm_dataset)
from src.summarizers.concept_based_ilp import summarize_single_doc, summarize_cstnews

from datasets import load_dataset

if __name__ == '__main__':

    # test_corpus_name = 'temario'
    # test_corpus_name = 'cstnews'
    test_corpus_name = 'recognasumm'
    # test_corpus_name = 'xlsum'

    corpus_dir = f'../../data/corpora/{test_corpus_name}'
    summaries_dir = f'../../data/summaries/{test_corpus_name}'

    # min_ngram_size = 2
    # max_ngram_size = 2

    is_use_filtering = True
    is_save_json = False

    summary_size = 120

    # weighting_method = 'comb_mult'
    # weighting_method = 'sent_freq'
    # weighting_method = 'sent_pos'

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nCorpus: {test_corpus_name}')

    for min_ngram_size, max_ngram_size in [
        (1, 1),
        (2, 2),
        # (1, 2)
    ]:

        for weighting_method in [
            'comb_mult',
            # 'sent_freq',
            # 'sent_pos'
        ]:

            print(f'\nConfigurations: {min_ngram_size} -- {max_ngram_size} -- {is_use_filtering} '
                  f'-- {summary_size}')

            print(f'\nConcept Weighting Method: {weighting_method}\n')

            corpus = None
            summarize = None

            if test_corpus_name == 'cstnews':
                corpus = read_cstnews_corpus(summaries_dir)
                summarize = summarize_cstnews
            elif test_corpus_name == 'temario':
                corpus = read_temario_corpus(summaries_dir)
                summarize = summarize_single_doc
            elif test_corpus_name == 'xlsum':
                dataset = load_dataset('csebuetnlp/xlsum', 'portuguese')
                corpus_dict = read_xlsum_dataset(dataset)
                corpus = corpus_dict['test']
                summarize = summarize_single_doc
                is_save_json = True
            elif test_corpus_name == 'recognasumm':
                dataset = load_dataset('recogna-nlp/recognasumm')
                corpus_dict = read_recognasumm_dataset(dataset)
                corpus = corpus_dict['test']
                summarize = summarize_single_doc
                is_save_json = True
            else:
                print(f'Error. corpus_name must be "cstnews", "temario", "recognasumm" or "xlsum".')
                exit(-1)

            summarize(corpus, min_ngram_size, max_ngram_size, is_use_filtering, weighting_method, summary_size,
                      summaries_dir, is_save_json)

            import time

            time.sleep(60)
