from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

from config.nlp_models import (
    TRAINSET_PERCENTAGE, 
    EVALSET_PERCENTAGE,
    DEFAULT_NER_LABEL
)


class NLPBaseDataset(ABC):
    def __init__(self, annotations_filepath: str,  *args, **kwargs):
        """ Base class for NER datasets

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.

        """
        self.trainset_percentage: float = TRAINSET_PERCENTAGE
        self.evalset_percentage: float = EVALSET_PERCENTAGE
        self.labels: Set[str] = set(DEFAULT_NER_LABEL)

        self._vocab: Set[str] = set()
        self._data: List[dict] = self.load_data(annotations_filepath)

    @abstractmethod
    def load_data(self, annotations_filepath, self_assign: bool = True) -> List[Union[Dict, Tuple]]:  # pragma: no cover
        """Load annotated data from json file.

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.
            self_assign (bool): If True, the loaded data will be assigned to the data attribute.

        Returns:
            None
        """
        pass
    
    @property
    def num_train_examples(self) -> int:
        """Property for the number of training examples

        Returns:
            int, the number of training examples
        """
        return int(len(self) * self.trainset_percentage)
    
    @property
    def num_eval_examples(self) -> int:
        """Property for the number of evaluation examples

        Returns:
            int, the number of evaluation examples
        """
        return int(len(self) * self.evalset_percentage)

    @property
    def data(self) -> List[Dict]:  # pragma: no cover
        """Property for the dataset

        Returns:
            List[Dict], the annotated data
        """
        if self._data is None:
            raise ValueError('Dataset is not initialized.')

        return self._data

    def split_dataset(self, data: Optional[List] = None) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset to training set and evaluation set

        Args:
            data (Optional[List[Dict]], optional): The dataset to be split. Defaults to None.

        Returns:
            Tuple[List[Dict], List[Dict]] --> trainset, evalset
        """
        data = data if data is not None else self.data
        trainset_index: int = round(len(data) * self.trainset_percentage)

        return data[:trainset_index], data[trainset_index:]

    def __len__(self) -> int:
        """Property for the length of the dataset

        Returns:
            int, the length of the dataset
        """
        return len(self.data)
