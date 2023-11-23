import json
from typing import Dict, List, Set, Tuple

import spacy
from loguru import logger
from spacy.tokens import DocBin

from src.NLP.nlp_base import NLPBaseDataset
from src.schemas.datatypes import RawSpacyData
from src.tools.general_tools import (
    get_filepath,
    dump_pickled_data
)
from src.tools.text_tools import exclude_list_duplicates, preprocess_text


class SpacyDataset(NLPBaseDataset):
    def __init__(self, annotations_filepath: str, *args, **kwargs) -> None:
        """Prepare data for spacy models.

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.
        """
        super().__init__(annotations_filepath, *args, **kwargs)
    
    @property
    def vocab(self) -> Set[str]:
        return self._vocab

    def load_data(self, annotations_filepath, self_assign: bool = True) -> List[Dict]:
        """Load annotated data from json file.

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.
            self_assign (bool): If True, the loaded data will be assigned to the data attribute.

        Returns:
            None
        """
        with open(annotations_filepath, 'r') as fin:
            data = list()

            for line in fin:
                # load json line with annotations
                entry: Dict = json.loads(line)

                # Remove unwanted symbols (e.g. leftovers from table conversion "||")
                entry = preprocess_text(entry)

                # exclude duplicates
                entry['label']: List = exclude_list_duplicates(entry['label'])

                data.append(entry)

        if self_assign:
            self._data = data

        return data

    def to_spacy_format(self, base_path: str) -> None:  # pragma: no cover
        """
        Args:
            base_path (str): The destination folder for the dataset.

        Returns
        """
        def to_spacy_format_dataset(dataset: List[Dict]) -> Tuple[DocBin, int, int]:  # pragma: no cover
            """Convert a list of entries to spacy format for a specific dataset.

            Args:
                dataset (List[Dict]): The annotated data.

            Returns:
                Tuple[DocBin, int, int] --> spacy_dataset, num of accepted annotations, num of skipped annotations
            """
            nlp = spacy.blank('en')
            doc_bin = DocBin()

            # keep track of the number of valid and skipped annotations
            n_skipped = n_valid = 0
            for file in dataset:
                ents = list()

                doc = nlp(file['data'])
                self.vocab.update([token.text for token in doc])

                for annotation in exclude_list_duplicates(file['label']):
                    # make sure the span is valid: start = annotation[0], end = annotation[1], label = annotation[2]
                    span = doc.char_span(annotation[0], annotation[1], annotation[2], alignment_mode='strict')

                    # add entity to labels
                    self.labels.add(annotation[2])

                    if not span:
                        n_skipped += 1
                        continue

                    n_valid += 1
                    ents.append(span)
                doc.ents = ents
                doc_bin.add(doc)

            return doc_bin, n_valid, n_skipped

        # split dataset
        trainset, evalset = self.split_dataset()

        # create trainset doc bins
        doc_bin_train, n_val_train, n_skip_train = to_spacy_format_dataset(trainset)
        logger.info(f'There are {n_val_train} valid and {n_skip_train} skipped entries in train set.')

        # create evalset doc bins
        doc_bin_eval, n_val_eval, n_skip_eval = to_spacy_format_dataset(evalset)
        logger.info(f'There are {n_val_eval} valid and {n_skip_eval} skipped entries in eval set.')

        # create dataset with the total number of annotated data
        doc_bin_all, _, _ = to_spacy_format_dataset(self.data)

        # store the data in the spacy format
        doc_bin_train.to_disk(get_filepath(base_path, 'train.spacy'))
        doc_bin_eval.to_disk(get_filepath(base_path, 'eval.spacy'))
        doc_bin_all.to_disk(get_filepath(base_path, 'dataset.spacy'))

        # store the raw data
        dump_pickled_data(get_filepath(base_path, 'evalset.pkl'), evalset)
        dump_pickled_data(get_filepath(base_path, 'trainset.pkl'), trainset)
        dump_pickled_data(get_filepath(base_path, 'nervaluator_evalset.pkl'), self._to_evaluator_format(evalset))

        # store the labels
        dump_pickled_data(get_filepath(base_path, 'labels.pkl'), list(self.labels))

    @staticmethod
    def _to_evaluator_format(dataset: List[Dict[str, str]]) -> List[RawSpacyData]:
        """Converts the dataset to the format that the nervaluator expects.

        Args:
            dataset (List[Dict[str, str]]): The dataset to be converted. Each entry should contain
                                            a 'label' key which points to a list of start, end, tag labels.

        Returns:
            List[List[Dict]]
        """
        def get_label_mapping(file: Dict) -> List[Dict]:
            """
            Args:
                  file (Dict): The file with the annotated labels.

            Returns:
                List[Dict], list with the labels
            """
            labels = list()

            for label in file['label']:
                # start = label[0], end = label[1], tag = label[2]
                start, end, tag = label[0], label[1], label[2]
                labels.append({
                    'label': tag,
                    'start': start,
                    'end': end
                })
            return labels

        ground_truths: List[List[Dict]] = [get_label_mapping(file) for file in dataset]

        return ground_truths
