import os
import json


if __name__ == '__main__':

    #corpus_name = 'temario'
    corpus_name = 'cstnews'
    # corpus_name = 'xlsum_pt'

    summ_model_name = 'gpt_neo_br'
    #summ_model_name = 'alpaca_lora'
    #summ_model_name = 'cabrita_lora'

    temperature = 0.3

    summaries_dir = f'/IFES/Projeto Mestrado/Datasets/' \
                    f'summaries/{corpus_name}'

    summaries_json_dir = f'/IFES/Projeto Mestrado/Datasets/' \
                         f'summaries/{corpus_name}_json'

    print(f'\nCorpus: {corpus_name}')

    summaries_file_path = os.path.join(summaries_json_dir, f'{summ_model_name}_{temperature}.json')

    summaries_corpus = {}

    if os.path.exists(summaries_file_path):
        with open(file=summaries_file_path, mode='r', encoding='utf-8') as file:
            summaries_corpus = json.load(file)

    for document_name, summaries_dict in summaries_corpus.items():

        document_name = document_name.lower()

        print(f'\nDocument: {document_name}')

        summaries_document_path = os.path.join(summaries_dir, document_name)

        for summary_name, summary_text in summaries_dict.items():

            print(f'  {summary_name}: {summary_text}')

            summary_path = os.path.join(summaries_document_path, f'{summary_name}.txt')

            with open(file=summary_path, mode='w', encoding='utf-8') as file:
                file.write(summary_text)
