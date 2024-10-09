import torch
import numpy as np
import evaluate 
import os
import nltk
import time

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.fine_tuning.fine_tuning_helper import preprocess_function


nltk.download('punkt')
rouge = evaluate.load('rouge')


def compute_eval_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ['\n'.join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result = rouge.compute(predictions=decoded_preds,
                           references=decoded_labels,
                           use_stemmer=False)
    result = {key: value for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result['gen_len'] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}


if __name__ == '__main__':

    is_turn_off_computer = False

    # model_name = 'ptt5_small'
    # model_name = 'ptt5_base'
    # model_name = 'ptt5_large'

    # model_name = 'flan_t5_small'
    # model_name = 'flan_t5_base'
    model_name = 'flan_t5_large'

    # model_name = 'ptt5_v2_small'
    # model_name = 'ptt5_v2_base'
    # model_name = 'ptt5_v2_large'

    dataset_name = 'recognasumm'
    # dataset_name = 'xlsum'

    use_fp16 = False

    models_dir = '../../data/models'
    training_dir = '../../data/training'

    n_examples = -1

    num_epochs = 20

    max_input_len = 512
    max_summary_len = 150

    batch_size = 32

    if model_name == 'flan_t5_base' or model_name == 'ptt5_v2_base':
        batch_size = 8
    elif model_name == 'flan_t5_large':
        batch_size = 3
    elif '_large' in model_name:
        batch_size = 4

    model_checkpoint = None

    if model_name == 'flan_t5_small':
        model_checkpoint = 'google/flan-t5-small'
    elif model_name == 'flan_t5_base':
        model_checkpoint = 'google/flan-t5-base'
    elif model_name == 'flan_t5_large':
        model_checkpoint = 'google/flan-t5-large'
    elif model_name == 'ptt5_small':
        model_checkpoint = 'unicamp-dl/ptt5-small-portuguese-vocab'
    elif model_name == 'ptt5_base':
        model_checkpoint = 'unicamp-dl/ptt5-base-portuguese-vocab'
    elif model_name == 'ptt5_large':
        model_checkpoint = 'unicamp-dl/ptt5-large-portuguese-vocab'
    elif model_name == 'ptt5_v2_small':
        model_checkpoint = 'unicamp-dl/ptt5-v2-small'
    elif model_name == 'ptt5_v2_base':
        model_checkpoint = 'unicamp-dl/ptt5-v2-base'
    elif model_name == 'ptt5_v2_large':
        model_checkpoint = 'unicamp-dl/ptt5-v2-large'
    else:
        print(f'\nError. Model Name {model_name} not found!')
        exit(-1)

    model_path = os.path.join(models_dir, f'{model_name}_{dataset_name}')

    output_dir = f'{training_dir}/{model_name}_{dataset_name}'

    os.makedirs(model_path, exist_ok=True)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device} -- Use FP16: {use_fp16} -- Batch size: {batch_size} -- '
          f'Turn Off Computer: {is_turn_off_computer}')

    print(f'\nModel: {model_name} -- {model_checkpoint}')

    if dataset_name == 'xlsum':
        dataset = load_dataset('csebuetnlp/xlsum', 'portuguese')
    elif dataset_name == 'recognasumm':
        dataset = load_dataset("recogna-nlp/recognasumm")
        dataset = dataset.rename_column("index", "id")
        dataset = dataset.rename_column("Noticia", "text")
        dataset = dataset.rename_column("Sumario", "summary")
    else:
        print(f'\nError. DATASET Name {dataset_name} Invalid!')
        exit(-1)

    # dataset = dataset.filter(lambda example: len(example['summary'].split()) >= 25)

    if n_examples > 0:
        train_dataset = dataset['train'].select(range(n_examples))
        validation_dataset = dataset['validation'].select(range(n_examples))
    else:
        train_dataset = dataset['train']
        validation_dataset = dataset['validation']

    print(f'\nTrain: {len(train_dataset)}')
    print(f'Validation: {len(validation_dataset)}\n')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, legacy=False)

    train_encoded_dataset = train_dataset.map(
        preprocess_function, batched=True, fn_kwargs={
            'max_input_len_': max_input_len, 'max_target_len_': max_summary_len,
            'tokenizer_': tokenizer})

    validation_encoded_dataset = validation_dataset.map(
        preprocess_function, batched=True, fn_kwargs={
            'max_input_len_': max_input_len, 'max_target_len_': max_summary_len,
            'tokenizer_': tokenizer})

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    logging_eval_steps = len(train_encoded_dataset) // batch_size

    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        eval_steps=logging_eval_steps,
        logging_steps=logging_eval_steps,
        evaluation_strategy='epoch',
        predict_with_generate=True,
        save_total_limit=1,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',
        greater_is_better=True,
        push_to_hub=False,
        fp16=use_fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_encoded_dataset,
        eval_dataset=validation_encoded_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_eval_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5
            )
        ]
    )

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.evaluate()

    trainer.save_model(model_path)

    print('\n\n***Finetunning Complete!***')

    if is_turn_off_computer:
        print('\nTurning off computer ...')
        time.sleep(2 * 60)
        os.system('shutdown -h now')
