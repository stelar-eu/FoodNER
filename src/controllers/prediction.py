from typing import Dict, List

from transformers import pipeline

from src.controllers import NERDataLoader
from src.NER.utils import PredictionUtils
from src.NER.spacy.base_model import SpacyModel
from src.NER.bert.bert_hf_model import BertTokenClassifierHF
from src.tools.general_tools import load_json_data
from src.schemas.datatypes import BertPipelineData, PipelineEntry, RawSpacyData, SpacyEntry
from src.tools.text_tools import extract_text_for_predict


class ModelPredict(NERDataLoader):
    def __init__(self, dataset_format: str, model_format: str) -> None:  # pragma: no cover
        """Predict NER labels.

        Args:
            dataset_format (str): The format that freetext will be exported. Default value spacy.
            model_format (str): The model to train and evaluate. Default value rnn.
        """
        super().__init__(dataset_format, model_format)

    def predict(self) -> None:
        """Predict NER labels for the given inference sentences.

        Returns:
            None
        """
        # load inference dataset
        documents: List = load_json_data(self._inference_filepath)

        if self.model_format in ['transition', 'transition_simple', 'transformer']:

            if self.model_format == 'transition':
                spacy_config = 'spacy_config_ner+tok2vec'

            elif self.model_format == 'transition_simple':
                spacy_config = 'spacy_config'

            else:
                spacy_config = 'spacy_config_transformer'

            self._spacy_predict(spacy_config, documents)

        elif self.model_format == 'bert':
            self._bert_predict(documents)

        else:
            raise ValueError('The given dataset format type is not valid')

    def _spacy_predict(self, spacy_config: str, documents: List[Dict]) -> None:
        """Method Executes the prediction inference with spacy.

        Args:
            spacy_config (str): The configuration for the spacy model.
            documents (List[Dict]): The documents with the raw text.

        Returns:
            None
        """
        spacy_model = SpacyModel(
            dataset_base_path=self._dataset_base_path,
            output_dir=self._model_base_path,
            config_type=spacy_config,
            mode='train'
        )
        for doc in documents:

            doc_id: str = doc['id']
            texts: List = extract_text_for_predict(doc['data'])

            # extract predictions
            predictions: List[RawSpacyData] = spacy_model.predict(texts, as_dict=True)

            # parse and post results
            utils = PredictionUtils(doc_id, texts, predictions)
            utils.parse_and_post_results()

    def _bert_predict(self, documents: List[Dict]) -> None:
        """Method Executes the prediction inference for bert.

        Args:
            documents (List[Dict]): The documents with the raw text.

        Returns:
            None
        """
        def bert_pipeline_to_raw_spacy_format(bert_pipeline: PipelineEntry) -> SpacyEntry:
            return {
                'start': bert_pipeline['start'],
                'end': bert_pipeline['end'],
                'label': bert_pipeline['entity_group'],
                'text': bert_pipeline['word']
            }

        bert_model = BertTokenClassifierHF(
            dataset_base_path=self._dataset_base_path,
            eval_base_path=self._model_base_path,
            mode='train',
        )
        model_path = bert_model.get_latest_ckpt_path()
        classifier = pipeline('ner', model=model_path)

        for doc in documents:

            doc_id: str = doc['id']
            texts: List[str] = extract_text_for_predict(doc['data'])

            # extract predictions
            predictions: List[BertPipelineData] = classifier(texts, aggregation_strategy='simple')

            # bring to spacy format
            predictions: List[RawSpacyData] = [
                [bert_pipeline_to_raw_spacy_format(entry) for entry in text_predictions]
                for text_predictions in predictions
            ]

            # parse and post results
            utils = PredictionUtils(doc_id, texts, predictions)
            utils.parse_and_post_results()
