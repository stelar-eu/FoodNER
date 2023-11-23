import os
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk, DatasetDict
from nervaluate import Evaluator
from loguru import logger

from config.nlp_models import (
    BERT_EVAL_KWARGS,
    BERT_IGNORE_INDEX,
    BERT_MODEL_NAME,
    BERT_TRAIN_KWARGS,
    DEFAULT_DEVICE,
    DEFAULT_NER_LABEL
)
from src.tools.general_tools import dump_json_data, load_pickled_data, get_filepath


class BertTokenClassifierHF:
    """Bert classifier class.

    Args:
        dataset_base_path (str): The path of evaluation results.
        eval_base_path (str): The path of evaluation results.
        model_name (str): The model name.
        mode (str): The mode of the bert_hf class, train or eval.
        device (str): The selected devise, CPU or GPU.
    """
    def __init__(
        self, 
        dataset_base_path: str,
        eval_base_path: str,
        model_name: str = BERT_MODEL_NAME,
        mode: str = 'train',
        device: Optional[str] = None
    ) -> None:
        self._model = None
        self._tokenizer = None
        self.mode = mode
        self.dataset_base_path: str = dataset_base_path
        self.model_name: str = model_name
        self.dataset: DatasetDict = load_from_disk(dataset_base_path)
        self.output_dir: str = eval_base_path
        self.device: str = device if device else DEFAULT_DEVICE
        self.label_names: List[str] = load_pickled_data(os.path.join(dataset_base_path, 'labels.pkl'))
        self.label2id: Dict = {label: idx for idx, label in enumerate(self.label_names)}
        self.id2label: Dict = {idx: label for idx, label in enumerate(self.label_names)}

    @property
    def model(self):  # pragma: no cover
        """Load pretrained bert model"""
        if self._model is None:
            logger.info(f'Using {self.device} to initialize pretrained HF model.')

            self._model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, 
                num_labels=len(self.label_names),
                id2label=self.id2label, 
                label2id=self.label2id
            ).to(self.device)

        return self._model
    
    @property
    def tokenizer(self):  # pragma: no cover
        """bert tokenization from pretrained model."""
        if self._tokenizer is None:
            self._tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        return self._tokenizer
    
    @property
    def training_args(self):  # pragma: no cover
        """Training arguments for the model."""
        if self.mode == 'train':
            train_kwargs = BERT_TRAIN_KWARGS
        else:
            train_kwargs = BERT_EVAL_KWARGS

        return TrainingArguments(
            output_dir=self.output_dir,
            no_cuda=self.device == 'cpu',
            **train_kwargs,
        )

    def train(self) -> None:  # pragma: no cover
        """Methods train the bert model in the dataset."""
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        logger.info(f'Started training model (checkpoints will be saved under {self.output_dir})')
        if self.mode == 'train':
            train_ds = self.dataset['all']
            eval_ds = None
            compute_metrics_fn = None

        else:
            train_ds = self.dataset['train']
            eval_ds = self.dataset['eval']
            compute_metrics_fn = self._computer_nervaluate_metrics

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics_fn
        )

        trainer.train()    

    def _computer_nervaluate_metrics(self, eval_pred: Tuple[torch.Tensor, List[List[int]]]) -> Dict:
        """Method computes the evaluation metrics for the model.

        Args:
            eval_pred(Tuple): A tuple with the evaluation predictions

        Returns:
              Dict
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=2)

        true_predictions = [
            [self.label_names[pred] for (pred, lab) in zip(prediction, label) if lab != BERT_IGNORE_INDEX]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_names[lab] for (_, lab) in zip(prediction, label) if lab != BERT_IGNORE_INDEX]
            for prediction, label in zip(predictions, labels)
        ]
        # get a list of all tags without I- and B- prefixes
        tags = list(set([tag[2:] for tag in self.label_names if tag != DEFAULT_NER_LABEL]))

        # use nervaluate to evaluate the model
        evaluator = Evaluator(true_labels, true_predictions, tags=tags, loader='list')
        results, results_per_tag = evaluator.evaluate()

        eval_metrics: Dict[str, float] = {
            'precision': results['strict']['precision'],
            'recall': results['strict']['recall'],
            'f1': results['strict']['f1']
        }

        for tag in results_per_tag:
            eval_metrics[f'{tag}_precision'] = results_per_tag[tag]['strict']['precision']
            eval_metrics[f'{tag}_recall'] = results_per_tag[tag]['strict']['recall']
            eval_metrics[f'{tag}_f1'] = results_per_tag[tag]['strict']['f1']

        if self.mode != 'train':
            # store evaluation results (will overwrite previous results)
            dump_json_data(path=get_filepath(self.output_dir, 'eval_results.json'), data=results)
            dump_json_data(path=get_filepath(self.output_dir, 'eval_results_per_tag.json'), data=results_per_tag)

        return eval_metrics

    def predict_token_classes(self, text: str) -> List[str]:  # pragma: no cover
        """Predicts token classes for a given text. The output list contains
           the predicted token class for each token in the text.

        Args:
            text (str): Text to predict token classes for (in raw form)

        Returns:
            List[str]: List of predicted token classes.
            WARNING: the length of the list is equal to the number of tokens + 2 - for the special start and end tokens.
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=2)
        predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]

        return predicted_token_class
    
    def evaluate(self) -> Dict:  # pragma: no cover
        """Evaluates the model on the evaluation dataset by using the huggingface evaluator.
           There is nothing wrong with this method, but we still prefer to use the nervaluate
           library for evaluation since it can be used easily regardless of the model type.

        Returns:
            Dict: Dictionary containing the evaluation metrics.
        """
        self._use_finetuned_model()
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=self._computer_nervaluate_metrics,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            eval_dataset=self.dataset['eval']
        )
        return trainer.evaluate()

    def get_latest_ckpt_path(self) -> str:
        """Get the latest checkpoint path.

        Returns:
            str, the path
        """
        all_checkpoints = [ckpt for ckpt in os.listdir(self.output_dir) if 'checkpoint' in ckpt]

        if len(all_checkpoints) == 0:
            raise ValueError('No checkpoints found in output directory. Please train model first.')

        latest_checkpoint = sorted(all_checkpoints, key=lambda x: int(x.split('checkpoint-')[-1]))[-1]
        logger.info(f'Using latest checkpoint: {latest_checkpoint}.')

        return get_filepath(self.output_dir, latest_checkpoint)

    def _use_finetuned_model(self, model_path: str = None) -> None:  # pragma: no cover
        """Utilize the fine-tuned bert model.

        Args:
             model_path(str): The path in the trained model.

        Returns:
            None
        """

        if model_path is None:
            model_path = self.get_latest_ckpt_path()

        if not os.path.isdir(model_path):
            raise ValueError(f'Model path {model_path} is not a checkpoint directory.'
                             f'Please specify a valid checkpoint directory or train a new model.')

        self._model = AutoModelForTokenClassification \
            .from_pretrained(model_path, num_labels=len(self.label_names)) \
            .to(self.device)
