import os
import sys
import torch
from dotenv import load_dotenv

from src.corpora.corpora_utils import read_cstnews_corpus, read_temario_corpus
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from tqdm import tqdm


if __name__ == '__main__':

    load_dotenv()

    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    # model_name = 'llama31_8b_it'
    # model_name = 'mistralv2_7b'
    model_name = 'gemma2_9b_it'

    test_corpus_name = 'temario'
    # test_corpus_name = 'cstnews'

    corpus_dir = f'../../data/corpora/{test_corpus_name}'
    summaries_dir = f'../../data/summaries/{test_corpus_name}'

    min_summary_len = 80
    max_summary_len = 150

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nModel: {model_name}')

    test_dataset = None

    if test_corpus_name == 'temario':
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
        for doc in corpus.list_cluster_documents:
            full_text = ' '.join([doc.text for doc in doc.documents]).replace('\n', ' ').strip()
            test_dataset.append(
                {
                    'name': doc.name,
                    'text': full_text,
                    'reference_summary': doc.abstractive_reference_summaries
                }
            )
    else:
        print(f'Erro. Corpus Option Invalid!')
        exit(-1)

    print(f'\nTest Corpus: {test_corpus_name} -- {len(test_dataset)}\n')

    if model_name == 'llama31_8b_it':
        model_checkpoint = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif model_name == 'gemma2_9b_it':
        model_checkpoint = 'google/gemma-2-9b-it'
    elif model_name == 'mistralv2_7b':
        model_checkpoint = 'mistralai/Mistral-7B-Instruct-v0.2'
    else:
        print(f'Erro. Model Option Invalid!')
        exit(-1)

    template = """
    Escreva um resumo em PORTUGUÊS DO BRASIL para o artigo de notícias a seguir com no MÁXIMO {MAX} palavras.
    ARTIGO: ```{TEXTO}```.
    """

    template = template.replace('{MAX}', str(max_summary_len))

    print(f'Template: {template}')

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, quantization_config=quantization_config,
        device_map='auto')

    with (tqdm(total=len(test_dataset), file=sys.stdout, colour='red', desc='Summarizing') as pbar):

        for example in test_dataset:

            document_dir = os.path.join(summaries_dir, example['name'].lower())

            os.makedirs(document_dir, exist_ok=True)

            summary_path = os.path.join(document_dir, f'{model_name}.txt')

            if os.path.exists(summary_path):
                pbar.update(1)
                continue

            prompt = template.replace('{TEXTO}', example['text'])

            messages = [
                {
                    'role': 'user',
                    'content': prompt
                }
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors='pt',
                max_length=2048
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids('<|eot_id|>')
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=150,
                min_length=min_summary_len,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                remove_invalid_values=True,
                eos_token_id=terminators,
            )

            summary_logits = outputs[0][input_ids.shape[-1]:]

            generated_summary = tokenizer.decode(summary_logits, skip_special_tokens=True)

            generated_summary = generated_summary.replace('\n', ' ').strip()

            with open(file=summary_path, mode='w', encoding='utf-8') as file:
                file.write(generated_summary)

            pbar.update(1)
