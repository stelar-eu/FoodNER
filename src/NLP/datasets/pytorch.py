import json
from typing import List, Dict, Tuple, Set

from tqdm import tqdm

from config.nlp_models import (
    PT_END_TOKEN,
    PT_MAX_SEQUENCE_LENGTH,
    PT_PAD_TOKEN,
    PT_START_TOKEN,
    LSTM_TOKENIZER_TYPE,
    DEFAULT_NER_LABEL
)
from src.NLP.nlp_base import NLPBaseDataset
from src.NLP.tokenizers.nltk import NLTKTokenizer
from src.NLP.tokenizers.spacy import SpacyTokenizer
from src.tools.general_tools import (
    get_filepath,
    dump_pickled_data
)
from src.tools.text_tools import (
    exclude_list_duplicates,
    preprocess_text,
    tokens_to_ids
)


class PytorchProcessor(NLPBaseDataset):
    def __init__(
        self, 
        annotations_filepath: str,
        pad_token: str = PT_PAD_TOKEN,
        start_token: str = PT_START_TOKEN,
        end_token: str = PT_END_TOKEN,
        max_length: int = PT_MAX_SEQUENCE_LENGTH,
        tokenizer_type: str = LSTM_TOKENIZER_TYPE,
        *args, **kwargs
    ) -> None:
        """Prepare data for spacy models

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.
            pad_token (str): Token to use for padding.
            start_token (str): Token to use for start of sentence. Defaults to bert's [CLS] token.
            end_token (str): Token to use for end of sentence. Defaults to bert's [SEP] token.
            max_length (int): The maximum length of a sequence. Defaults to 512.
            tokenizer_type (str): The type of tokenizer to use. Defaults to 'spacy'.
        """
        self.start_token: str = start_token
        self.end_token: str = end_token
        self.pad_token: str = pad_token
        self.max_length: int = max_length

        self._word2idx: Dict[str, int] = {}
        self._idx2word: Dict[int, str] = {}
        self._label2idx: Dict[str, int] = {}
        self._idx2label: Dict[int, str] = {}

        if tokenizer_type == 'spacy':
            self.tokenizer = SpacyTokenizer()
        elif tokenizer_type == 'nltk':
            self.tokenizer = NLTKTokenizer()

        super().__init__(annotations_filepath, *args, **kwargs)
    
    @property
    def word2idx(self) -> Dict[str, int]:
        """Map words to ids.

        Returns:
            Dict[str, int], dict with words (key) and ids (value)
        """
        self._check_initialized()
        if not self._word2idx:
            self._word2idx = {token: idx for idx, token in enumerate(self.vocab)}

        return self._word2idx
    
    @property
    def vocab(self) -> Set[str]:
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: Set[str]) -> None:
        self._vocab = vocab
    
    @property
    def label2idx(self) -> Dict[str, int]:
        """Map labels to ids.

        Returns:
            Dict[str, str], dict with labels (key) and ids (value)
        """
        self._check_initialized()
        if not self._label2idx:
            self._label2idx = {label: idx for idx, label in enumerate(self.labels)}

        return self._label2idx
    
    @property
    def idx2word(self) -> Dict[int, str]:
        self._check_initialized()
        if not self._idx2word:
            self._idx2word = {idx: token for token, idx in self.word2idx.items()}

        return self._idx2word
    
    @property
    def idx2label(self) -> Dict[int, str]:
        self._check_initialized()
        if not self._idx2label:
            self._idx2label = {idx: label for label, idx in self.label2idx.items()}

        return self._idx2label
    
    def _check_initialized(self) -> None:
        """Check if the dataset is initialized."""
        if not self._data:
            raise ValueError('Dataset is not initialized. Make sure to call load_data() first.')
    
    def load_data(self, annotations_filepath, self_assign: bool = True) -> List[Tuple[List[str], List[str]]]:
        """Load annotated data from json file.

        Args:
            annotations_filepath (str): Path to the jsonl file that contains the annotated data.
            self_assign (bool): If True, the loaded data will be assigned to the data attribute.

        Returns:
            List[Dict]: The loaded data. A list of {'tokens': List[str], 'label': List[str]}
        """
        def convert_single_entry(file: Dict) -> Tuple[List[str], List[str]]:
            """Convert a single entry to pytorch format.

            Args:
                file (Dict): The annotated data. A single line of the original jsonl file.

            Returns:
                Tuple[List[str], List[str]] --> the tokenized text of the file, the extracted labels
            """
            spans = self.tokenizer.span_tokenize(file['data'])
            tokenized_text = [file['data'][span[0]:span[1]] for span in spans]

            # update the set with the vocabulary
            self.vocab.update(tokenized_text)
            new_labels = [DEFAULT_NER_LABEL] * len(tokenized_text)

            for annotation in file['label']:
                # start = annotation[0], end = annotation[1], label = annotation[2]
                char_start, char_end, label = annotation[0], annotation[1], annotation[2]

                for ii, span in enumerate(spans):
                    token_start, token_end = span
                    if (char_start <= token_start <= char_end) and (char_start <= token_end <= char_end):
                        new_labels[ii] = label
                        self.labels.add(label)

            return tokenized_text, new_labels

        with open(annotations_filepath, 'r') as fin:
            data = list()

            for line in tqdm(fin):
                # load json line with annotations
                entry = json.loads(line)
                # Remove unwanted symbols (e.g. leftovers from table conversion "||")
                entry = preprocess_text(entry)
                # exclude duplicates
                entry['label']: List = exclude_list_duplicates(entry['label'])
                # get the data in pytorch format (tokenized)
                entry = convert_single_entry(entry)
                data.append(entry)

        if self_assign:
            self._data = data

        return data
    
    def to_pt_format(self, base_path: str) -> None:
        """Convert the data to pytorch format.

        Args:
            base_path (str): The destination folder for the dataset.
        
        Returns:
            None
        """
        # convert to ids
        data: List[Tuple[List, List]] = [
            (tokens_to_ids(self.word2idx, token), tokens_to_ids(self.label2idx, label))
            for token, label in self.data
        ]
        # split into train, eval and test
        train_set, evaluation_set = self.split_dataset(data)
        # save the data
        dump_pickled_data(get_filepath(base_path, 'train.pkl'), train_set)
        dump_pickled_data(get_filepath(base_path, 'eval.pkl'), evaluation_set)
        # save the vocab and labels
        dump_pickled_data(get_filepath(base_path, 'vocabulary.pkl'), self.vocab)
        dump_pickled_data(get_filepath(base_path, 'labels.pkl'), self.labels)
        # save the index dictionaries
        dump_pickled_data(get_filepath(base_path, 'word2idx.pkl'), self.word2idx)
        dump_pickled_data(get_filepath(base_path, 'label2idx.pkl'), self.label2idx)
