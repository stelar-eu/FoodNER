import os


class Controller:
    """The general controller class."""

    def __init__(self, dataset_format: str) -> None:  # pragma: no cover
        """Initializes a controller class.

        Args:
            dataset_format (str): The format that freetext will be exported.
        """
        self.dataset_format: str = dataset_format

        # set dataset directories
        self._annotations_filepath = os.path.join('data', 'trainset', 'annotations.jsonl')
        self._inference_filepath = os.path.join('data', 'inference', 'raw_texts.jsonl')
        self._dataset_base_path: str = os.path.join('results', 'dataset', dataset_format)
        self._evaluation_base_path: str = os.path.join('results', 'evaluation', dataset_format)
        self._model_base_path: str = os.path.join('results', 'model', dataset_format)
        self._ner_base_path: str = os.path.join('results', 'ner', dataset_format)
        self._prediction_base_path: str = os.path.join('results', 'prediction', dataset_format)
        self._statistical_analysis_base_path: str = os.path.join('results', 'statistical_analysis', dataset_format)


class Singleton(type):
    """Singleton metaclass."""
    _instances = {}

    def __call__(cls, *args, **kwargs):  # pragma: no cover
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RawDataLoader(Controller, metaclass=Singleton):
    """Used by controllers that use the unprocessed data."""

    def __init__(self, data_from: str, data_until: str, dataset_format: str) -> None:  # pragma: no cover
        """Initializes a raw data loader class.

        Args:
            data_from (str): Starting date of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            data_until (str): Last day of the retrieved data from DataAPI. Input date format YYYY-MM-DD.
            dataset_format (str): The format that freetext will be exported. Default value spacy.
        """
        super().__init__(dataset_format)
        self.data_from: str = data_from
        self.data_until: str = data_until


class NERDataLoader(Controller, metaclass=Singleton):
    """Used by controllers that use the processed data."""

    def __init__(self, dataset_format: str, model_format: str) -> None:  # pragma: no cover
        """Initializes a raw data loader class.

        Args:
            dataset_format (str): The format that freetext will be exported. Default value spacy.
            model_format (str): The model to train and evaluate. Default value rnn.
        """
        super().__init__(dataset_format)
        self.model_format: str = model_format
