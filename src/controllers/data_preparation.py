from src.controllers import RawDataLoader
from src.etl.extraction.text_documents import TextExtractor
from src.NLP.datasets.bert import BertProcessor
from src.NLP.datasets.spacy import SpacyDataset
from src.NLP.datasets.pytorch import PytorchProcessor


class DataPreparation(RawDataLoader):

    def __init__(self, data_from: str, data_until: str, dataset_format: str) -> None:  # pragma: no cover
        """Prepare data for NLP models.

        Args:
            data_from (str): Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            data_until (str): Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            dataset_format (str): The format that freetext will be exported. Default value spacy.
        """
        super().__init__(data_from, data_until, dataset_format)
        self.lang: str = 'en'

    def prepare_train_dataset(self) -> None:
        """Prepare train dataset for NLP models.

        Returns:
            None
        """
        if self.dataset_format == 'spacy':
            spacy_ds = SpacyDataset(self._annotations_filepath)
            # extract dataset to spacy format type
            spacy_ds.to_spacy_format(self._dataset_base_path)

        elif self.dataset_format == 'pytorch':
            pt_ds = PytorchProcessor(self._annotations_filepath)
            # extract dataset to pytorch format type
            pt_ds.to_pt_format(self._dataset_base_path)
        
        elif self.dataset_format == 'bert':
            bert_ds = BertProcessor(self._annotations_filepath)
            # extract dataset to bert format type
            bert_ds.to_bert_format(self._dataset_base_path)

        else:
            raise ValueError('The given dataset format type is not valid')

    def prepare_inference_dataset(self) -> None:
        """Prepare inference dataset for NLP models.

        Returns:
            None
        """
        if self.dataset_format == 'spacy':
            text_extractor = TextExtractor(self.data_from, self.data_until, self.lang)
            text_extractor.corpus(self._inference_filepath)

        elif self.dataset_format == 'pytorch':
            raise NotImplementedError(f'Dataset format {self.dataset_format}  is not implemented yet.')

        else:
            raise ValueError('The given dataset format type is not valid')
