from loguru import logger

from src.controllers import NERDataLoader
from src.NER.spacy.base_model import SpacyModel
from src.NER.bert.bert_hf_model import BertTokenClassifierHF
from src.tools.general_tools import get_filepath


class ModelTraining(NERDataLoader):
    def __init__(self, dataset_format: str, model_format: str) -> None:  # pragma: no cover
        """Prepare data for NER models.

        Args:
            dataset_format (str): The format that freetext will be exported. Default value spacy.
            model_format (str): The model to train and evaluate. Default value rnn.
        """
        super().__init__(dataset_format, model_format)

    def train_ner(self) -> None:
        """Train NER models on the given training dataset.

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
                output_dir=self._model_base_path,
                config_type=spacy_config,
                mode='train'
            )
            logger.info('Start training spacy model')
            # extract dataset to spacy format type
            spacy_model.train()

        elif self.model_format == 'bert':
            bert_model = BertTokenClassifierHF(
                dataset_base_path=get_filepath(self._dataset_base_path, 'all'),
                eval_base_path=self._model_base_path,
                mode='train',
            )
            logger.info('Start training BERT HF model')
            bert_model.train()

        else:
            raise ValueError('The given dataset format type is not valid')
