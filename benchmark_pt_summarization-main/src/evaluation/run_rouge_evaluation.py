import sys

from src.corpora.corpora_utils import (read_cstnews_corpus, read_xsum_corpus, read_temario_corpus,
                                       read_recognasumm_dataset, read_summaries, read_summaries_json)
from datasets import load_dataset
from tqdm import tqdm
from src.basic_classes.document import Document
from src.evaluation_measures import rouge_parser


if __name__ == '__main__':

    # corpus_name = 'temario'
    corpus_name = 'cstnews'
    # corpus_name = 'recognasumm'
    # corpus_name = 'xlsum'

    corpus_path = f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/pt_summ_benchmark/' \
                  f'corpora/{corpus_name}'

    summaries_dir = (f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/pt_summ_benchmark/'
                     f'2024_08_22_summaries/summaries/{corpus_name}')

    is_save_json = False
    test_corpus = None

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

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
    limit_words = 150

    if is_save_json:
        read_summaries_json(test_corpus, summaries_dir)
    else:
        read_summaries(test_corpus, summaries_dir)

    with tqdm(total=len(test_corpus.documents), file=sys.stdout, colour='green', desc='Evaluating') as pbar:
        for document in test_corpus.documents:
            rouge_parser.evaluate_summaries(document, is_use_stemming, limit_words)
            pbar.update(1)

    rouge_parser.generate_report(test_corpus, summaries_dir)
