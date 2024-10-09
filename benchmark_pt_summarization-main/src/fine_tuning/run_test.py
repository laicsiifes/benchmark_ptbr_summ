import sys
import torch
import os
import json

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.corpora.corpora_utils import read_cstnews_corpus, read_temario_corpus
from more_itertools import batched


def tokenize_text(examples, max_input_len_, tokenizer_):
    model_inputs = tokenizer_(examples['text'], max_length=max_input_len_, padding=True, truncation=True,
                              return_tensors='pt')
    return model_inputs


if __name__ == '__main__':

    # model_name = 'ptt5_small'
    # model_name = 'ptt5_base'
    # model_name = 'ptt5_large'

    # model_name = 'flan_t5_small'
    # model_name = 'flan_t5_base'
    model_name = 'flan_t5_large'

    # model_name = 'ptt5_v2_small'
    # model_name = 'ptt5_v2_base'
    # model_name = 'ptt5_v2_large'

    train_dataset_name = 'recognasumm'
    # train_dataset_name = 'xlsum'

    # test_corpus_name = 'temario'
    # test_corpus_name = 'cstnews'
    test_corpus_name = 'recognasumm'
    # test_corpus_name = 'xlsum'

    models_dir = f'../../data/models/{model_name}_{train_dataset_name}'

    corpus_dir = f'../../data/corpora/{test_corpus_name}'
    summaries_dir = f'../../data/summaries/{test_corpus_name}'

    n_examples = -1
    batch_size = 32

    max_input_len = 512

    min_summary_len = 80
    max_summary_len = 150

    num_beams = 5

    if '_large' in model_name:
        batch_size = 8

    is_save_json = False

    os.makedirs(summaries_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    print(f'\nModel: {model_name} -- {models_dir}')

    test_dataset = None

    if test_corpus_name == 'xlsum':
        dataset = load_dataset('csebuetnlp/xlsum', 'portuguese')
        test_dataset = []
        for example in dataset['test']:
            test_dataset.append(
                {
                    'name': example['id'],
                    'text': example['text'],
                    'reference_summary': example['summary']
                }
            )
        is_save_json = True
    elif test_corpus_name == 'temario':
        corpus = read_temario_corpus(corpus_dir)
        test_dataset = []
        for docs in corpus.documents:
            test_dataset.append(
                {
                    'name': docs.name,
                    'text': docs.text,
                    'reference_summary': docs.reference_summary
                }
            )        
    elif test_corpus_name == 'cstnews':
        corpus = read_cstnews_corpus(corpus_dir)
        test_dataset = []
        for docs in corpus.list_cluster_documents:
            full_text = ' '.join([doc.text for doc in docs.documents])
            test_dataset.append(
                {
                    'name': docs.name,
                    'text': full_text,
                    'reference_summary': docs.abstractive_reference_summaries
                }
            )
    elif test_corpus_name == 'recognasumm':
        dataset = load_dataset('recogna-nlp/recognasumm')
        test_dataset = []
        for example in dataset['test']:
            test_dataset.append(
                {
                    'name': example['index'],
                    'text': example['Noticia'],
                    'reference_summary': example['Sumario']
                }
            )
        is_save_json = True
    else:
        print(f'Erro. Corpus Option Invalid!')
        exit(-1)

    if n_examples > 0:
        test_dataset = test_dataset[:n_examples]

    print(f'\nTest Corpus: {test_corpus_name} -- {len(test_dataset)}\n')

    tokenizer = AutoTokenizer.from_pretrained(models_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(models_dir)

    model = model.to(device)

    test_batches = [test_batch for test_batch in batched(iterable=test_dataset, n=batch_size)]

    list_documents_json = []

    with (tqdm(total=len(test_batches), file=sys.stdout, colour='red', desc='Summarizing') as pbar):

        for test_batch in test_batches:

            list_article_names = [t['name'] for t in test_batch]
            list_full_texts = [t['text'] for t in test_batch]
            list_reference_summaries = [t['reference_summary'] for t in test_batch]

            inputs = tokenizer(list_full_texts, max_length=max_input_len, padding=True, truncation=True,
                               return_tensors='pt')

            inputs_ids = inputs.input_ids.to(device)

            outputs = model.generate(inputs_ids, num_beams=num_beams, min_length=min_summary_len,
                                     max_length=max_summary_len, num_return_sequences=1, no_repeat_ngram_size=3,
                                     remove_invalid_values=True)

            list_generated_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for article_name, full_text, ref_summary, generated_summary in zip(
                    list_article_names, list_full_texts, list_reference_summaries, list_generated_summaries):
                if is_save_json:
                    article_dict = {
                        'name': article_name,
                        'text': full_text,
                        'reference_summary': ref_summary,
                        'generated_summary': generated_summary
                    }
                    list_documents_json.append(article_dict)
                    summary_path = os.path.join(summaries_dir, f'{model_name}_{train_dataset_name}.json')
                    json_object = json.dumps(list_documents_json, indent=4)
                    with open(file=summary_path, mode='w', encoding='utf-8') as outfile:
                        outfile.write(json_object)
                else:
                    document_dir = os.path.join(summaries_dir, str(article_name).lower())
                    os.makedirs(document_dir, exist_ok=True)
                    summary_path = os.path.join(document_dir,  f'{model_name}_{train_dataset_name}.txt')
                    with open(file=summary_path, mode='w', encoding='utf-8') as file:
                        file.write(generated_summary)

            pbar.update(1)
