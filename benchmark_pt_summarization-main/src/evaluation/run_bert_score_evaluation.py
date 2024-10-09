import torch
import sys

from src.corpora.corpora_utils import (read_cstnews_corpus, read_recognasumm_dataset, read_temario_corpus,
                                       read_summaries, read_xlsum_dataset, read_summaries_json)
from bert_score import BERTScorer
from tqdm import tqdm
from datasets import load_dataset
from src.basic_classes.document import Document
from src.evaluation_measures.bert_score_utils import evaluate_summaries_bert_score
from src.evaluation_measures.rouge_parser import generate_report


if __name__ == '__main__':

    # corpus_name = 'temario'
    corpus_name = 'cstnews'
    # corpus_name = 'recognasumm'
    # corpus_name = 'xlsum'

    corpus_path = f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/pt_summ_benchmark/' \
                  f'corpora/{corpus_name}'

    summaries_dir = (f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/pt_summ_benchmark/'
                     f'2024_08_22_summaries/summaries/{corpus_name}')

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

    test_corpus = None
    is_save_json = False

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
    elif corpus_name == 'xlsum':
        dataset = load_dataset('csebuetnlp/xlsum', 'portuguese')
        corpus_dict = read_xlsum_dataset(dataset)
        test_corpus = corpus_dict['test']
        is_save_json = True
    elif corpus_name == 'recognasumm':
        dataset = load_dataset("recogna-nlp/recognasumm")
        corpus_dict = read_recognasumm_dataset(dataset)
        test_corpus = corpus_dict['test']
        is_save_json = True
    else:
        print(f'Error. corpus_name must be "cstnews" or "xlsum_pt".')
        exit(-1)

    print(f'\n  Total documents: {len(test_corpus.documents)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    print(f'\nEvaluating\n')

    bert_scorer = BERTScorer(lang='pt', device=device)

    if is_save_json:
        read_summaries_json(test_corpus, summaries_dir)
    else:
        read_summaries(test_corpus, summaries_dir)

    with tqdm(total=len(test_corpus.documents), file=sys.stdout, colour='green', desc='  Evaluating') as pbar:
        for document in test_corpus.documents:
            evaluate_summaries_bert_score(document, bert_scorer)
            pbar.update(1)

    generate_report(test_corpus, summaries_dir)
