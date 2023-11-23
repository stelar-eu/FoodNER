from typing import Dict

import torch

TRAINSET_PERCENTAGE: float = 0.9
EVALSET_PERCENTAGE: float = 0.1

DEFAULT_NER_LABEL: str = "O"

# mappings from raw strings to remove to their raw lengths
SYMBOLS_TO_REMOVE = {'\|\|': 2, '\n': 1, '\t': 1, '\r': 1}

# Spacy constant variables
SPACY_BATCH_SIZE: int = 128

# PyTorch constant variables
PT_MAX_SEQUENCE_LENGTH: int = 512
PT_PAD_TOKEN: str = '[PAD]'
PT_START_TOKEN: str = '[CLS]'
PT_END_TOKEN: str = '[SEP]'
PT_BATCH_SIZE: int = 16
PT_LEARNING_RATE: float = 9e-5
PT_NUM_EPOCHS: int = 20

# BERT constant variables
BERT_MODEL_NAME: str = 'bert-base-cased'  # or uncased
BERT_GRADIENT_ACCUMULATION_STEPS: int = 2
BERT_MODEL_SEED: int = 1111
BERT_IGNORE_INDEX: int = -100  # Do not change this value
DEFAULT_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_EVAL_KWARGS: Dict = {
    'evaluation_strategy': 'epoch',  # steps might be too expensive
    'gradient_accumulation_steps': BERT_GRADIENT_ACCUMULATION_STEPS,
    'learning_rate': PT_LEARNING_RATE,
    'per_device_train_batch_size': PT_BATCH_SIZE,
    'per_device_eval_batch_size': PT_BATCH_SIZE,
    'num_train_epochs': PT_NUM_EPOCHS,
    'logging_steps': 1000,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'save_strategy': 'epoch',
    'seed': BERT_MODEL_SEED,
    'data_seed': BERT_MODEL_SEED,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',  # make sure this matches the compute_metric output
    'greater_is_better': True,
    # set to values from 1-10 if you can't accumulate all evaluation tensors on GPU
    # 'eval_accumulation_steps': 1
}
BERT_TRAIN_KWARGS: Dict = {
    'evaluation_strategy': 'no',  # steps might be too expensive
    'do_eval': False,
    'gradient_accumulation_steps': BERT_GRADIENT_ACCUMULATION_STEPS,
    'learning_rate': PT_LEARNING_RATE,
    'per_device_train_batch_size': PT_BATCH_SIZE,
    'per_device_eval_batch_size': PT_BATCH_SIZE,
    'num_train_epochs': PT_NUM_EPOCHS,
    'logging_steps': 1000,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'save_strategy': 'epoch',
    'seed': BERT_MODEL_SEED,
    'data_seed': BERT_MODEL_SEED
}

# LSTM Model constant variables
LSTM_TOKENIZER_TYPE: str = 'spacy'
