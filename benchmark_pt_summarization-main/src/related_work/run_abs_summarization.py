import os
import torch

from src.corpora.corpora_utils import (read_cstnews_corpus, read_temario_corpus, read_xlsum_dataset,
                                       read_recognasumm_dataset)
from src.summarizers.abstractive_summarizers import summarize_cstnews, summarize_single_doc
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset


if __name__ == '__main__':

    # test_corpus_name = 'temario'
    # test_corpus_name = 'cstnews'
    test_corpus_name = 'recognasumm'
    # test_corpus_name = 'xlsum'

    # model_name = 'abstractive_recognasumm'
    model_name = 'abstractive_recogna_xlsum'

    corpus_dir = f'../../data/corpora/{test_corpus_name}'
    summaries_dir = f'../../data/summaries/{test_corpus_name}'

    max_input_len = 512

    min_summary_len = 80
    max_summary_len = 150

    num_beams = 5

    is_save_json = False

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nCorpus: {test_corpus_name}')

    corpus = None
    summarize = None
    model_path = None

    if test_corpus_name == 'cstnews':
        corpus = read_cstnews_corpus(corpus_dir)
        summarize = summarize_cstnews
    elif test_corpus_name == 'temario':
        corpus = read_temario_corpus(corpus_dir)
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

    if model_name == 'abstractive_recognasumm':
        model_path = 'recogna-nlp/ptt5-base-summ'
    elif model_name == 'abstractive_recogna_xlsum':
        model_path = 'recogna-nlp/ptt5-base-summ-xlsum'
    else:
        print(f'Error. model_name must be "abstractive_recognasumm" or "recogna_xlsum".')
        exit(-1)

    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    print(f'\nSummarization Model: {model_name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    print(f'\nConfigurations: {max_input_len} -- {min_summary_len} -- {max_summary_len} -- '
          f'{num_beams}\n')

    model = model.to(device)

    summarize(corpus, max_input_len, min_summary_len, max_summary_len, num_beams, tokenizer, model,
              model_name, summaries_dir, device, is_save_json)
