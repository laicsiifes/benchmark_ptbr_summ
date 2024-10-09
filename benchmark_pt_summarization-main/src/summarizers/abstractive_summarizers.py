import torch
import sys
import json
import os

from tqdm import tqdm
from transformers import T5Tokenizer, PreTrainedModel

from src.basic_classes.corpus import Corpus
from src.basic_classes.summary import Summary
from src.corpora.corpora_utils import save_summary


def summarize_cstnews(corpus: Corpus, max_input_len: int, min_summary_len: int, max_summary_len: int,
                      num_beams: int, tokenizer: T5Tokenizer, model: PreTrainedModel,
                      model_name: str, summaries_dir: str, device: torch.device, is_save_json: bool = False):

    with tqdm(total=len(corpus.list_cluster_documents), file=sys.stdout, colour='red',
              desc='Summarizing') as pbar:

        for cluster_documents in corpus.list_cluster_documents:

            full_text = ' '.join([doc.text for doc in cluster_documents.documents])

            inputs_ids = tokenizer.encode(full_text, max_length=max_input_len, truncation=True,
                                          return_tensors='pt')

            inputs_ids = inputs_ids.to(device)

            tokens_summary_ids = model.generate(inputs_ids, min_length=min_summary_len,
                                                max_length=max_summary_len,
                                                num_beams=num_beams, no_repeat_ngram_size=3,
                                                early_stopping=True)

            summary = tokenizer.decode(tokens_summary_ids[0], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)

            summary = Summary(name=model_name, text=summary)

            save_summary(cluster_documents.name, summary, summaries_dir)

            pbar.update(1)


def summarize_single_doc(corpus: Corpus, max_input_len: int, min_summary_len: int, max_summary_len: int,
                         num_beams: int, tokenizer: T5Tokenizer, model: PreTrainedModel,
                         model_name: str, summaries_dir: str, device: torch.device, is_save_json: bool):

    list_documents_json = []

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='red', desc='Summarizing') as pbar:

        for document in corpus.documents:

            inputs_ids = tokenizer.encode(document.text, max_length=max_input_len, truncation=True,
                                          return_tensors='pt')
            inputs_ids = inputs_ids.to(device)
            tokens_summary_ids = model.generate(inputs_ids, min_length=min_summary_len,
                                                max_length=max_summary_len, num_beams=num_beams,
                                                no_repeat_ngram_size=3, early_stopping=True)
            summary = tokenizer.decode(tokens_summary_ids[0], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
            summary = Summary(name=model_name, text=summary)

            if is_save_json:
                article_dict = {
                    'name': document.name,
                    'generated_summary': summary.text
                }
                list_documents_json.append(article_dict)
                summary_path = os.path.join(summaries_dir, f'{model_name}.json')
                json_object = json.dumps(list_documents_json, indent=4)
                os.makedirs(summaries_dir, exist_ok=True)
                with open(file=summary_path, mode='w', encoding='utf-8') as outfile:
                    outfile.write(json_object)
            else:
                save_summary(document.name, summary, summaries_dir)

            pbar.update(1)
