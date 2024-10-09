from src.corpora.corpora_utils import (metrics_dataset_cstnews, metrics_dataset_recognasumm, metrics_dataset_temario, read_cstnews_corpus, read_temario_corpus)
from datasets import load_dataset

if __name__ == '__main__':

    #corpus_name = 'temario'
    #corpus_name = 'cstnews'
    corpus_name = 'recognasumm'

    corpus_path = f'D:/IFES/Projeto Mestrado/nlp_files/corpora/{corpus_name}'
    metrics_path = 'D:/IFES/Projeto Mestrado/nlp_files/corpora/'
    if corpus_name == 'cstnews':
        corpus = read_cstnews_corpus(corpus_path, )

        metrics_dataset_cstnews(corpus, metrics_path, corpus_name)

    elif corpus_name == 'temario':
        corpus = read_temario_corpus(corpus_path)

        metrics_dataset_temario(corpus, metrics_path, corpus_name)

    elif corpus_name == 'recognasumm':

        dataset = load_dataset('recogna-nlp/recognasumm')

        metrics_dataset_recognasumm(dataset, metrics_path, corpus_name)
        
    else:
        print(f'Error. {corpus_name} INVALID!.')
        exit(-1)

