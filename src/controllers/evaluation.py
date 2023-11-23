from loguru import logger

from src.controllers import NERDataLoader
from src.NER.spacy.base_model import SpacyModel
from src.NER.bert.bert_hf_model import BertTokenClassifierHF


class ModelEvaluation(NERDataLoader):
    def __init__(self, dataset_format: str, model_format: str) -> None:  # pragma: no cover
        """Prepare data for NER models.

        Args:
            dataset_format (str): The format that freetext will be exported. Default value spacy.
            model_format (str): The model to train and evaluate. Default value rnn.
        """
        super().__init__(dataset_format, model_format)

    def evaluate_ner_evaluation_set(self) -> None:
        """Evaluate NER models on the given training and evaluation datasets.

        Returns:
            None
        """
        if self.model_format in ['transition', 'transition_simple', 'transformer']:

            if self.model_format == 'transition':
                spacy_config = 'spacy_config_ner+tok2vec'

            elif self.model_format == 'transition_simple':
                spacy_config = 'spacy_config'

            else:
                spacy_config = 'spacy_config_transformer'

            spacy_model = SpacyModel(
                dataset_base_path=self._dataset_base_path,
                output_dir=self._evaluation_base_path,
                config_type=spacy_config,
                mode='eval'
            )
            logger.info('Start training spacy model')
            # extract dataset to spacy format type
            spacy_model.train()

            logger.info('Evaluating on the evaluation set')
            spacy_model.evaluate()

        elif self.model_format == 'bert':
            bert_model = BertTokenClassifierHF(
                dataset_base_path=self._dataset_base_path,
                eval_base_path=self._evaluation_base_path,
                mode='eval',
            )
            logger.info('Start training BERT HF model')
            bert_model.train()

        else:
            raise ValueError('The given dataset format type is not valid')
