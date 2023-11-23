import os
from typing import Dict, List, Optional, Union

import spacy
from loguru import logger
from spacy.cli.train import train as spacy_train
from nervaluate import Evaluator

from config.nlp_models import SPACY_BATCH_SIZE, DEFAULT_NER_LABEL
from src.schemas.datatypes import PredictedFormat, RawSpacyData
from src.tools.general_tools import dump_json_data, get_filepath, load_pickled_data


class SpacyModel:
    def __init__(
        self,
        dataset_base_path: str,
        output_dir: str,
        config_type: Optional[str] = 'spacy_config',
        mode: str = 'train',
        batch_size: int = SPACY_BATCH_SIZE,
    ) -> None:
        """Prepare data for NER models.

        Args:
            dataset_base_path (str): The path of the evaluation results.
            output_dir (str): The destination path of the results.
            batch_size (str): The batch size.
        """
        self.batch_size: int = batch_size
        self.config_path: str = get_filepath('config', f'{config_type}_{mode}.cfg')
        self.evalset_data_path: str = get_filepath(dataset_base_path, 'eval.spacy')
        self.trainset_data_path: str = get_filepath(dataset_base_path, 'train.spacy')

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f'Config path {self.config_path} does not exist.')

        if not os.path.isfile(self.evalset_data_path):
            raise FileNotFoundError(f'Eval data path {self.evalset_data_path} does not exist.')

        if not os.path.isfile(self.trainset_data_path):
            raise FileNotFoundError(f'Train data path {self.trainset_data_path} does not exist.')

        self.dataset_base_path: str = dataset_base_path
        self.output_dir: str = output_dir
        self.mode: str = mode
        self._model = None
    
    @property
    def model(self):  # pragma: no cover
        """Load the trained model."""
        if not self._model:
            try:
                self._model = spacy.load(get_filepath(self.output_dir, 'model-best'))

            except OSError as ose:
                logger.error(f'Could not load model from {self.output_dir}. Make sure you call train() first.')
                raise ose

        return self._model

    def train(self) -> None:  # pragma: no cover
        """Train rnn model with Spacy framework."""
        overrides: Dict = {
            'paths.train': self.trainset_data_path,
            'nlp.batch_size': self.batch_size,
            'paths.dev': self.evalset_data_path,
        }

        if self.mode == 'train':
            logger.info('Detected training-only mode with spacy model. This means that we will use '
                        'an evaluation set that is a subset of the training set and we will evaluate '
                        'on it only at the end of training.')
        logger.info('Training model...')

        spacy_train(
            self.config_path,
            overrides=overrides,
            output_path=self.output_dir
        )
        logger.info(f'Spacy model saved under {self.output_dir} (model-best is the model with the best F1-score'
                    f'on the evaluation set), and model-last refers to the model\'s last epoch\'s checkpoint.')

    def evaluate(self) -> Dict:  # pragma: no cover
        """Evaluate the model on the evaluation set. Notice that the exact same train/eval splitting
           process is conducted as in the original case where the model was first trained.

        Returns:
            None
        """
        # load raw evaluation set
        evalset: RawSpacyData = load_pickled_data(
            get_filepath(self.dataset_base_path, 'evalset.pkl')
        )
        # load raw ground truths (annotations) for nervaluator evaluation
        ground_truths: List[RawSpacyData] = load_pickled_data(
            get_filepath(self.dataset_base_path, 'nervaluator_evalset.pkl')
        )
        # load all distinct annotation labels
        all_tags: List[str] = load_pickled_data(
            get_filepath(self.dataset_base_path, 'labels.pkl')
        )
        # evaluation data for nervaluate
        pred_eval_data: List[RawSpacyData] = self.predict(
            [file['data'] for file in evalset],
            as_dict=True
        )

        results, results_per_tag = Evaluator(ground_truths, pred_eval_data, tags=list(all_tags)).evaluate()

        # store results
        dump_json_data(path=get_filepath(self.output_dir, 'eval_results.json'), data=results)
        dump_json_data(path=get_filepath(self.output_dir, 'eval_results_per_tag.json'), data=results_per_tag)

        return results

    def predict(
        self,
        texts: Union[List[str], str],
        as_dict: bool = False
    ) -> List[Union[RawSpacyData, PredictedFormat]]:  # pragma: no cover
        """Predict entities in text.

        Args:
            texts (List|str): The text (or list of texts) for which we want to predict entities.
            as_dict (bool): If true return a dictionary, otherwise return a list
                            of tuples (first entry is the text second is the entity label).

        Returns:
            List[RawSpacyData]: A list of dictionaries containing the entity sub text, the entity label,
                                and the start and end character ids.
            List[PredictedFormat]: A list of tuples containing the entity sub text and the entity label.
        """
        predictions = []
        if isinstance(texts, str):
            texts = [texts]

        for txt in texts:
            predictions.append(self._to_predict(txt, as_dict=as_dict))

        return predictions
    
    def _to_predict(
        self,
        text: str,
        as_dict: bool = False
    ) -> Union[RawSpacyData, PredictedFormat]:
        """Predict entities in text.

        Args:
            text (str): The text for which we want to predict entities.
            as_dict (bool): If true return a dictionary, otherwise return a list
                            of tuples (first entry is the text second is the entity label).

        Returns:
            PredictedFormat: A tuple with the entity sub text and the entity label.
            RawSpacyData: A dictionary with the entity sub text, the entity label, and the start and end character ids.
        """
        doc = self.model(text)
        predictions = []
        for ent in doc.ents:

            if as_dict is False:
                predictions.append((ent.text, ent.label_))

            elif ent.label_ != DEFAULT_NER_LABEL:
                predictions.append({
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text
                })

        return predictions
