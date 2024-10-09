def preprocess_function(examples, max_input_len_, max_target_len_, tokenizer_):
    model_inputs = tokenizer_(examples['text'], max_length=max_input_len_, truncation=True)
    labels = tokenizer_(examples['summary'], max_length=max_target_len_, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
